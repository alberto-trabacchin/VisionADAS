import json
import os
import shutil
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import argparse
import cv2


# Desired combination
desired_combination = {
    'pedestrian': 5000,  # Minimum area in pixels for 'pedestrian'
    'bicycle': 5000,  # Minimum area in pixels for 'car',
    'rider': 5000
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help='.../bdd100k/')
    parser.add_argument("--output-path", type=str, help='.../bdd100k/custom_dataset/')
    parser.add_argument("--img_size", type=int, default=600)
    args = parser.parse_args()
    return args


def prepare_custom_dataset(annotation_path, img_dir, output_dir, size):
    # Create output directories
    class_0_dir = os.path.join(output_dir, 'class_0')
    class_1_dir = os.path.join(output_dir, 'class_1')
    os.makedirs(class_0_dir, exist_ok=True)
    os.makedirs(class_1_dir, exist_ok=True)

    # Load annotations
    with open(annotation_path, 'r') as file:
        data = json.load(file)

    pbar = tqdm(total=len(data))
    # Process each image
    for item in data:
        image_name = item['name']
        if 'labels' not in item:
            pbar.update(1)
            continue
        classes_in_image = set()
        valid_for_class_1 = False

        for annotation in item['labels']:
            if 'category' in annotation and 'box2d' in annotation:
                class_name = annotation['category']
                bbox = annotation['box2d']
                width = abs(bbox['x2'] - bbox['x1'])
                height = abs(bbox['y2'] - bbox['y1'])
                area = width * height

                # Check area against threshold if the class is part of the desired combination
                if class_name in desired_combination and area >= desired_combination[class_name]:
                    valid_for_class_1 = True
                classes_in_image.add(class_name)

        # Check for desired combination
        if any([cls in classes_in_image for cls in desired_combination]) and valid_for_class_1:
            # Copy to class 1
            class_dir = class_1_dir
        else:
            # Copy to class 0
            class_dir = class_0_dir
        
        # Construct source and destination paths
        src_path = os.path.join(img_dir, image_name)
        dst_path = os.path.join(class_dir, image_name)

        with Image.open(src_path) as img:
            # Resize the image to be square
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img.save(dst_path)
        pbar.update(1)
    pbar.close()


def prepare_unlabeled_set(vids_dir, output_dir, size):
    videos = list(vids_dir.glob('*.mov'))
    sample_fac = 10
    size = 600
    pbar = tqdm(total=len(videos))
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for v in videos:
        video_name = v.stem
        # Construct source and destination paths       
        cap = cv2.VideoCapture(str(v))
        frames = []
        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if counter % sample_fac == 0:
                frames.append(frame)
            counter += 1
        cap.release()

        # Resize the frames to be square
        for i, frame in enumerate(frames):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
            f_dir = output_dir / f"{video_name}_{i * sample_fac}.jpg"
            cv2.imwrite(str(f_dir), frame)
        pbar.update(1)
    pbar.close()



if __name__ == '__main__':
    args = parse_args()

    # Paths to the dataset
    train_annotation_file = Path(f'{args.data_path}/labels/det_20/det_train.json')
    val_annotation_file = Path(f'{args.data_path}/labels/det_20/det_val.json')
    train_image_directory = Path(f'{args.data_path}/images/100k/train/')
    val_image_directory = Path(f'{args.data_path}/images/100k/val/')
    output_directory = Path(f'{args.output_path}/')
    test_image_directory = Path(f'{args.data_path}/images/100k/test/')
    videos_directory = Path(f'{args.data_path}/videos/train/')

    # Run the dataset preparation
    print('Preparing custom train dataset...')
    prepare_custom_dataset(train_annotation_file, train_image_directory, output_directory/"train", args.img_size)
    print('Preparing custom validation dataset...')
    prepare_custom_dataset(val_annotation_file, val_image_directory, output_directory/"val", args.img_size)
    print('Preparing custom unlabeled dataset...')
    prepare_unlabeled_set(videos_directory, output_directory/"unlabeled", args.img_size)