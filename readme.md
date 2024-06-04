# VisionADAS

## Description
Here below the set of instructions for configuring BDD100k dataset. Download the zip files in the same folder. During the unzipping process, a parent folder named *bdd100k* will be created automatically.

## 1. Download BDD100k train images
Download train images from the following link
```bash
wget https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip
```

Unzip the downloaded file
```bash
unzip 100k_images_train.zip
```

## 2. Download BDD100k val images
Download val images from the following link
```bash
wget https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip
```

Unzip the downloaded file
```bash
unzip 100k_images_val.zip
```

## 3. Download BDD100k trainval labels
Download labels from the following link
```bash
wget https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip
```

Unzip the downloaded file
```bash
unzip bdd100k_det_20_labels_trainval.zip
```

## 4. Download BDD100k videos (only for MPL)
Download some train videos from the following link (each zip file contains 1000 videos). You can start with 1~4 zip files.

The train videos are named *bdd100k_videos_train_xx.zip*.

Each zip file is ~18GB.
```bash
http://dl.yf.io/bdd100k/video_parts/
```
Unzip each video file.

## 5. Prepare the dataset
To prepare the preprocessed dataset, run the following command:
```bash
python prepare_bdd100k.py \
       --data_dir=/path/to/bdd100k \
       --img_size=512
```
Custom dataset is created in *bdd100k/custom_dataset/*

## Tracking with Weights & Biases
To enable wandb:
```bash
wandb online
```
To disable wandb:
```bash
wandb disabled
```
If enabled, it will create a folder named *wandb*.
By default, models with minimum val loss are saved in the *checkpoints/* folder.

## Supervised Training
To run the default configuration:
```bash
python train_sl.py \
       --name="default_supervised" \
       --data-path=/path/to/bdd100k/folder \
       --img-size=512 \
       --patch-size=32 \
       --embed-dim=256 \
       --depth=12 \
       --heads=8 \
       --mlp-dim=2048 \
       --lr=1e-3 \
       --batch-size=32 \
       --epochs=100 \
       --workers=4 \
       --device=cuda \
       --seed=42 \
       --eta-min=1e-5
```
Specify the path to bdd100k in --data-path.

## Semi-Supervised Training (MPL)
To run the default configuration:
```bash
python train_mpl.py \
       --name="default_mpl" \
       --data-path=/path/to/bdd100k/folder \
       --img-size=512 \
       --patch-size=32 \
       --embed-dim=256 \
       --depth=12 \
       --heads=8 \
       --mlp-dim=2048 \
       --lr=1e-3 \
       --batch-size=16 \
       --epochs=100 \
       --workers=4 \
       --device=cuda \
       --seed=42 \
       --eta-min=1e-5
```
Specify the path to bdd100k in --data-path.