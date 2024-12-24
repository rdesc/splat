import os
import shutil
import cv2
import yaml
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Copy and optionally rectify selected images.")
    parser.add_argument('--save_dir', type=str, required=True, help="Target directory to save the selected images.")
    parser.add_argument('--ego_view', nargs='+', type=str, default=[], help="List of directories for ego view images.")
    parser.add_argument('--ego_keep_every', type=int, default=1, help="Interval to keep every nth ego view frame.")
    parser.add_argument('--overhead_view', nargs='+', type=str, default=[], help="List of directories for overhead view images.")
    parser.add_argument('--overhead_keep_every', type=int, default=2, help="Interval to keep every nth overhead view frame.")
    parser.add_argument('--rectify', action='store_true', help="Enable rectification of images.")
    parser.add_argument('--camera_intrinsic', type=str, help="Path to the camera intrinsic YAML file.")
    return parser.parse_args()

def load_camera_intrinsics(filename):
    if not os.path.isfile(filename):
        print(f"Intrinsic calibration for {filename} does not exist.")
        exit(3)
    with open(filename) as f:
        data = yaml.safe_load(f)
    intrinsics = {
        'K': np.array(data['camera_matrix']['data']).reshape(3, 3),
        'D': np.array(data['distortion_coefficients']['data']).reshape(1, 5),
        'R': np.array(data['rectification_matrix']['data']).reshape(3, 3),
        'P': np.array(data['projection_matrix']['data']).reshape((3, 4)),
        'distortion_model': data['distortion_model']
    }
    print(f"Loaded camera intrinsics for {filename}")
    return intrinsics

def rectify_images(image, intrinsics):
    height, width, _ = image.shape
    mapx, mapy = cv2.initUndistortRectifyMap(
        intrinsics['K'], intrinsics['D'], intrinsics['R'], intrinsics['P'],
        (width, height), cv2.CV_32FC1
    )
    return cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)

def copy_images(source_dirs, keep_every, destination, prefix_file_name='', rectify=False, intrinsics=None):
    if rectify and intrinsics is None:
        raise ValueError("Rectification enabled but no intrinsics provided.")

    count = 0
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory '{source_dir}' does not exist.")
            continue

        files = sorted(os.listdir(source_dir))
        for i, file_name in enumerate(files):
            if '.ini' in file_name:
                continue
            count += 1
            if count % keep_every == 0:
                source_path = os.path.join(source_dir, file_name)
                if os.path.isfile(source_path):
                    destination_path = os.path.join(destination, "input", f"{prefix_file_name}_{count}.png")

                    if rectify:
                        image = cv2.imread(source_path)
                        if image is not None:
                            rectified_image = rectify_images(image, intrinsics)
                            cv2.imwrite(destination_path, rectified_image)
                            print(f"Rectified and saved: {source_path} -> {destination_path}")
                        else:
                            print(f"Failed to load image: {source_path}")
                    else:
                        shutil.copy(source_path, destination_path)
                        print(f"Copied: {source_path} -> {destination_path}")

def main():
    args = parse_arguments()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "input"), exist_ok=True)

    intrinsics = None
    if args.rectify:
        if not args.camera_intrinsic:
            print("Camera intrinsic file must be provided when rectification is enabled.")
            exit(3)
        intrinsics = load_camera_intrinsics(args.camera_intrinsic)

    if args.ego_view:
        copy_images(
            args.ego_view, args.ego_keep_every, args.save_dir,
            prefix_file_name='egoview', rectify=args.rectify, intrinsics=intrinsics
        )

    if args.overhead_view:
        copy_images(
            args.overhead_view, args.overhead_keep_every, args.save_dir,
            prefix_file_name='overhead', rectify=args.rectify, intrinsics=intrinsics
        )

if __name__ == '__main__':
    main()