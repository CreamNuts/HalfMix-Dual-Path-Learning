"""
    References:
        https://github.com/hysts/pytorch_mpiigaze/blob/master/tools/preprocess_mpiifacegaze.py    
"""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm, trange


def mat448ToH5224(person_id: str, dataset_dir: Path, output_path: Path) -> None:
    with h5py.File(dataset_dir / f"{person_id}.mat", "r") as f_input:
        images = f_input.get("Data/data")[()]
        labels = f_input.get("Data/label")[()][:, :4]
    assert len(images) == len(labels) == 3000

    # N C H W -> N H W C and BGR -> RGB
    images = images.transpose(0, 2, 3, 1)[:, :, :, ::-1].astype(np.uint8)

    # 448x448 -> 224x224
    resize_images = []
    for image in tqdm(images, leave=True):
        resize_images.append(cv2.resize(image, (224, 224)))

    images = np.array(resize_images)
    poses = labels[:, 2:].astype(np.float32)
    gazes = labels[:, :2].astype(np.float32)

    with h5py.File(output_path, "a") as f_output:
        f_output.create_dataset(f"{person_id}/face_patch", data=images)
        f_output.create_dataset(f"{person_id}/face_head_pose", data=poses)
        f_output.create_dataset(f"{person_id}/face_gaze", data=gazes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "MPIIFaceGaze.h5"
    if output_path.exists():
        raise ValueError(f"{output_path} already exists.")

    dataset_dir = Path(args.data_dir)
    for person_id in trange(15):
        person_id = f"p{person_id:02}"
        mat448ToH5224(person_id, dataset_dir, output_path)


if __name__ == "__main__":
    main()
