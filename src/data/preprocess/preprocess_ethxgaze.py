"""
    Reference: https://phi-ai.buaa.edu.cn/Gazehub/
"""
import argparse
from pathlib import Path

import cv2
import h5py

from ..components.var import ETHXGAZE_TEST, ETHXGAZE_TRAIN, ETHXGAZE_VAL, get_num


def process_person(h5files_path, image_path, sub_id, anno_path, test):
    datas = h5py.File(h5files_path, "r")
    keys = (
        ["cam_index", "face_head_pose", "face_mat_norm", "face_patch", "frame_index"]
        if test
        else [
            "cam_index",
            "face_gaze",
            "face_head_pose",
            "face_mat_norm",
            "face_patch",
            "frame_index",
        ]
    )
    length = datas[keys[0]].shape[0]
    print(f"Processing.. {h5files_path}")
    print(f"==> Length: {length}")

    image_path.mkdir(parents=True, exist_ok=True)

    with anno_path.open("a") as outfile:
        for i in range(length):
            img = datas["face_patch"][i, :]
            imo_path = image_path / f"{sub_id}_{i}.jpg"
            if not imo_path.exists():
                imo_path.write_bytes(cv2.imencode(".jpg", img)[1].tobytes())
            head = ",".join(datas["face_head_pose"][i, :].astype("str"))
            norm_mat = ",".join(datas["face_mat_norm"][i, :].astype("str").flatten())
            cam_index = ",".join(datas["cam_index"][i, :].astype("str"))
            frame_index = ",".join(datas["frame_index"][i, :].astype("str"))
            if not test:
                gaze = ",".join(datas["face_gaze"][i, :].astype("str"))
                outfile.write(
                    f"{get_num(sub_id)} {i} {gaze} {head} {cam_index} {frame_index} {norm_mat}\n"
                )
            else:
                outfile.write(
                    f"{get_num(sub_id)} {i} {head} {cam_index} {frame_index} {norm_mat}\n"
                )
    datas.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    subdict = {"train": ETHXGAZE_TRAIN, "val": ETHXGAZE_VAL, "test": ETHXGAZE_TEST}
    for mode in ["train", "val", "test"]:
        if mode == "val":
            orig_path = data_path / "train"
        else:
            orig_path = data_path / mode

        if mode != "test":
            header = "subject index gaze head cam_index frame_index normmat\n"
            test = False
        else:
            header = "subject index head cam_index frame_index normmat\n"
            test = True

        image_path = data_path / "Image" / mode
        anno_path = data_path / "Label" / f"{mode}.label"
        image_path.mkdir(parents=True, exist_ok=True)
        anno_path.parent.mkdir(parents=True, exist_ok=True)

        with anno_path.open("w") as outfile:
            outfile.write(header)

        filenames = sorted(orig_path.iterdir())
        for _, filename in enumerate(filenames):
            sub_id = filename.stem
            if get_num(sub_id) in subdict[mode]:
                process_person(filename, image_path, sub_id, anno_path, test)


if __name__ == "__main__":
    main()
