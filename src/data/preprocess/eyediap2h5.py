"""
    References:
        http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#eyediap-eye    
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import data_processing_core as dpc
import h5py
import numpy as np
from tqdm import tqdm


def CamParamsDecode(path):
    cal = {}
    fh = open(path, "r")
    # Read the [resolution] section
    fh.readline().strip()
    cal["size"] = [int(val) for val in fh.readline().strip().split(";")]
    cal["size"] = cal["size"][0], cal["size"][1]
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(";")])
    cal["intrinsics"] = np.array(vals).reshape(3, 3)
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(";")])
    cal["R"] = np.array(vals).reshape(3, 3)
    # Read the [T] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(";")])
    cal["T"] = np.array(vals).reshape(3)
    fh.close()
    return cal


def preprocess_video(folder_path: Path):
    video_path = folder_path / "rgb_vga.mov"
    head_path = folder_path / "head_pose.txt"
    anno_path = folder_path / "eye_tracking.txt"
    camparams_path = folder_path / "rgb_vga_calibration.txt"
    target_path = folder_path / "screen_coordinates.txt"

    number = int(folder_path.name.split("_")[0])
    person = "p" + str(number)

    # Read annotations
    with open(head_path) as infile:
        head_info = infile.readlines()
    with open(anno_path) as infile:
        anno_info = infile.readlines()
    with open(target_path) as infile:
        target_info = infile.readlines()
    length = len(target_info) - 1

    # Read camera parameters
    cam_info = CamParamsDecode(camparams_path)
    camera = cam_info["intrinsics"]
    cam_rot = cam_info["R"]
    cam_trans = cam_info["T"] * 1000

    face_patches, lefteye_patches, righteye_patches = [], [], []
    face_gazes, face_head_poses = [], []
    # Read video
    cap = cv2.VideoCapture(str(video_path))
    for index in range(1, length + 1):
        ret, frame = cap.read()
        if (index - 1) % 15 != 0:
            continue

        # Calculate rotation and transition of head pose.
        head = head_info[index]
        head = list(map(eval, head.strip().split(";")))
        if len(head) != 13:
            # Error Head Pose
            continue

        head_rot = head[1:10]
        head_rot = np.array(head_rot).reshape([3, 3])
        head_rot = np.dot(cam_rot, head_rot)
        head1 = cv2.Rodrigues(head_rot)[0].T[0]
        head2d = dpc.HeadTo2d(head1)

        # rotate the head into camera coordinate system
        head_trans = np.array(head[10:13]) * 1000
        head_trans = np.dot(cam_rot, head_trans)
        head_trans = head_trans + cam_trans

        # Calculate the 3d coordinates of origin.
        anno = anno_info[index]
        anno = list(map(eval, anno.strip().split(";")))
        if len(anno) != 19:
            # Error annotation
            continue
        anno = np.array(anno)

        left3d = anno[13:16] * 1000
        left3d = np.dot(cam_rot, left3d) + cam_trans
        right3d = anno[16:19] * 1000
        right3d = np.dot(cam_rot, right3d) + cam_trans

        face3d = (left3d + right3d) / 2
        face3d = (face3d + head_trans) / 2

        left2d = anno[1:3]
        right2d = anno[3:5]

        # Calculate the 3d coordinates of target
        target = target_info[index]
        target = list(map(eval, target.strip().split(";")))
        if len(target) != 6:
            print("[Error target]")
            continue
        target3d = np.array(target)[3:6] * 1000

        # Normalize the left eye image
        norm = dpc.norm(
            center=face3d,
            gazetarget=target3d,
            headrotvec=head_rot,
            imsize=(224, 224),
            camparams=camera,
        )

        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Acquire essential info
        im_face = norm.GetImage(frame)
        gaze = norm.GetGaze(scale=False)
        head = norm.GetHeadRot(vector=False)
        head = cv2.Rodrigues(head)[0].T[0]

        gaze2d = dpc.GazeTo2d(gaze)
        head2d = dpc.HeadTo2d(head)

        # Crop Eye Image
        left2d = norm.GetNewPos(left2d)
        right2d = norm.GetNewPos(right2d)

        im_left = norm.CropEyeWithCenter(left2d)
        im_left = dpc.EqualizeHist(im_left)
        im_right = norm.CropEyeWithCenter(right2d)
        im_right = dpc.EqualizeHist(im_right)

        # Save the data
        face_patches.append(im_face)
        lefteye_patches.append(im_left)
        righteye_patches.append(im_right)
        face_gazes.append(gaze2d)
        face_head_poses.append(head2d)

    return dict(
        person=person,
        face_patch=face_patches,
        lefteye_patch=lefteye_patches,
        righteye_patch=righteye_patches,
        face_gaze=face_gazes,
        face_head_pose=face_head_poses,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "Eyediap.h5"
    if output_path.exists():
        raise ValueError(f"{output_path} already exists.")

    data_dict = defaultdict(lambda: defaultdict(list))

    dataset_dir = Path(args.data_dir)
    datafolder_dir = dataset_dir / "Data"
    subfolders = sorted(datafolder_dir.iterdir(), key=lambda x: int(x.name.split("_")[0]))
    print("Start to process data...")
    for subfolder in tqdm(subfolders):
        if "FT" not in subfolder.name:
            result = preprocess_video(subfolder)
            person = result.pop("person")
            for key, value in result.items():
                data_dict[person][key].extend(value)

    print("Finish processing data, and now start saving data...")

    with h5py.File(output_path, "w") as f_output:
        for person, data in tqdm(data_dict.items()):
            for key, value in data.items():
                if "patch" in key:
                    value = np.asarray(value).astype(np.uint8)
                else:
                    value = np.asarray(value).astype(np.float32)
                f_output[f"{person}/{key}"] = np.array(value)
    print("Finish saving data.")


if __name__ == "__main__":
    main()
