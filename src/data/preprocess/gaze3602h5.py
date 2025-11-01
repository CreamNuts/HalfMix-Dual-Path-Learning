"""
    Reference:
        http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#gaze360
"""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io as sio
from tqdm import trange


def GazeTo2d(gaze):
    yaw = np.arctan2(gaze[0], -gaze[2])
    pitch = np.arcsin(gaze[1])
    return np.array([pitch, yaw])


def CropFaceImg(img, head_bbox, cropped_bbox):
    bbox = np.array(
        [
            (cropped_bbox[0] - head_bbox[0]) / head_bbox[2],
            (cropped_bbox[1] - head_bbox[1]) / head_bbox[3],
            cropped_bbox[2] / head_bbox[2],
            cropped_bbox[3] / head_bbox[3],
        ]
    )

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    # Find the image center and crop head images with length = max(weight, height)
    center = np.array([bbox_pixel[0] + bbox_pixel[2] // 2, bbox_pixel[1] + bbox_pixel[3] // 2])

    length = int(max(bbox_pixel[2], bbox_pixel[3]) / 2)

    center[0] = max(center[0], length)
    center[1] = max(center[1], length)

    result = img[
        (center[1] - length) : (center[1] + length), (center[0] - length) : (center[0] + length)
    ]

    result = cv2.resize(result, (224, 224))
    return result


def CropEyeImg(img, head_bbox, cropped_bbox):
    bbox = np.array(
        [
            (cropped_bbox[0] - head_bbox[0]) / head_bbox[2],
            (cropped_bbox[1] - head_bbox[1]) / head_bbox[3],
            cropped_bbox[2] / head_bbox[2],
            cropped_bbox[3] / head_bbox[3],
        ]
    )

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    center = np.array([bbox_pixel[0] + bbox_pixel[2] // 2, bbox_pixel[1] + bbox_pixel[3] // 2])
    height = bbox_pixel[3] / 36
    weight = bbox_pixel[2] / 60
    ratio = max(height, weight)

    size = np.array([ratio * 30, ratio * 18]).astype("int")

    center[0] = max(center[0], size[0])
    center[1] = max(center[1], size[1])

    result = img[
        (center[1] - size[1]) : (center[1] + size[1]),
        (center[0] - size[0]) : (center[0] + size[0]),
    ]

    result = cv2.resize(result, (60, 36))
    return result


def preprocess_Gaze360(data_path: Path, output_path: Path):
    mat_dir = data_path / "metadata.mat"
    metadata = sio.loadmat(mat_dir)

    recordings = metadata["recordings"]
    gaze_dir = metadata["gaze_dir"]
    head_bbox = metadata["person_head_bbox"]
    face_bbox = metadata["person_face_bbox"]
    lefteye_bbox = metadata["person_eye_left_bbox"]
    righteye_bbox = metadata["person_eye_right_bbox"]
    splits = metadata["splits"]

    split_index = metadata["split"]
    recording_index = metadata["recording"]
    person_index = metadata["person_identity"]
    frame_index = metadata["frame"]

    total_num = recording_index.shape[1]

    faces = dict(train=[], val=[], test=[], unused=[])
    lefteyes = dict(train=[], val=[], test=[], unused=[])
    righteyes = dict(train=[], val=[], test=[], unused=[])
    gazes = dict(train=[], val=[], test=[], unused=[])

    for i in trange(total_num):
        im_path = (
            data_path
            / "imgs"
            / recordings[0, recording_index[0, i]][0]
            / "head"
            / f"{person_index[0, i]:06d}"
            / f"{frame_index[0, i]:06d}.jpg"
        )

        if (face_bbox[i] == np.array([-1, -1, -1, -1])).all():  # face is not detected
            continue

        category = splits[0, split_index[0, i]][0]  # train, val, test, unused

        img = cv2.imread(str(im_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = CropFaceImg(img, head_bbox[i], face_bbox[i])
        lefteye = CropEyeImg(img, head_bbox[i], lefteye_bbox[i])
        righteye = CropEyeImg(img, head_bbox[i], righteye_bbox[i])
        faces[category].append(face)
        lefteyes[category].append(lefteye)
        righteyes[category].append(righteye)

        gaze = gaze_dir[i]
        gaze2d = GazeTo2d(gaze)
        gazes[category].append(gaze2d)

    for category in ["train", "val", "test", "unused"]:
        faces[category] = np.asarray(faces[category]).astype(np.uint8)
        lefteyes[category] = np.asarray(lefteyes[category]).astype(np.uint8)
        righteyes[category] = np.asarray(righteyes[category]).astype(np.uint8)
        gazes[category] = np.asarray(gazes[category]).astype(np.float32)

        with h5py.File(output_path, "a") as f_output:
            f_output.create_dataset(f"{category}/face_patch", data=faces[category])
            f_output.create_dataset(f"{category}/lefteye_patch", data=lefteyes[category])
            f_output.create_dataset(f"{category}/righteye_patch", data=righteyes[category])
            f_output.create_dataset(f"{category}/face_gaze", data=gazes[category])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "Gaze360.h5"
    if output_path.exists():
        raise ValueError(f"{output_path} already exists.")

    dataset_dir = Path(args.data_dir)
    preprocess_Gaze360(dataset_dir, output_path)


if __name__ == "__main__":
    main()
