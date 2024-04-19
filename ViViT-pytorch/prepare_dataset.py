from pathlib import Path
import argparse
import tqdm
from PIL import Image
import shutil
from nuscenes_edit import NuScenes
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--dest-path", type=str)
    parser.add_argument("--img-size", type=int, default=400)
    parser.add_argument("--ann-path", type=str, default="./annotations/")
    parser.add_argument("--vid-len", type=int, default=190)
    parser.add_argument("--sample-period", type=int, default=4)
    args = parser.parse_args()
    return args


def get_vids_paths(nusc, annotations, mode):
    video_paths = {
        "safe": [],
        "dangerous": []
    }
    pbar = tqdm.tqdm(total=len(annotations), position=0, leave=True, desc=f"Reading {mode} data...")
    for ann in annotations:
        rec_paths = []
        scene_rec = nusc.get("scene", ann["token"])
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["CAM_FRONT"])
        has_more_frames = True
        while has_more_frames:
            img_path, _, _ = nusc.get_sample_data(sd_rec["token"])
            rec_paths.append(img_path)
            if not sd_rec["next"] == "":
                sd_rec = nusc.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        if ann["annotation_id"] == 0:
            video_paths["safe"].append(rec_paths)
        else:
            video_paths["dangerous"].append(rec_paths)
        pbar.update(1)
    pbar.close()
    return video_paths


def get_min_length(train_video_paths, val_video_paths, test_video_paths):
    min_len = min(
        min([len(vid) for vid in train_video_paths["safe"]]),
        min([len(vid) for vid in train_video_paths["dangerous"]]),
        min([len(vid) for vid in val_video_paths["safe"]]),
        min([len(vid) for vid in val_video_paths["dangerous"]]),
        min([len(vid) for vid in test_video_paths["safe"]]),
        min([len(vid) for vid in test_video_paths["dangerous"]])
    )
    return min_len


def make_dataset(args, video_paths, name):
    if Path(f"{args.dest_path}/{name}").exists():
        shutil.rmtree(f"{args.dest_path}/{name}")
    Path(f"{args.dest_path}/{name}/safe").mkdir(parents=True, exist_ok=True)
    Path(f"{args.dest_path}/{name}/dangerous").mkdir(parents=True, exist_ok=True)
    pbar = tqdm.tqdm(total=(len(video_paths["safe"]) + len(video_paths["dangerous"])), 
                     position=0, leave=True, desc=f"Creating {name} dataset...")
    for i, vid in enumerate(video_paths["safe"]):
        Path(f"{args.dest_path}/{name}/safe/{i}").mkdir(parents=True, exist_ok=True)
        vid = vid[:args.vid_len]
        vid = [vid[i] for i in range(0, len(vid), args.sample_period)]
        for j, frame in enumerate(vid):
            img = Image.open(frame)
            img = img.resize((args.img_size, args.img_size))
            img.save(f"{args.dest_path}/{name}/safe/{i}/{j}.jpg")
        pbar.update(1)
    for i, vid in enumerate(video_paths["dangerous"]):
        Path(f"{args.dest_path}/{name}/dangerous/{i}").mkdir(parents=True, exist_ok=True)
        vid = vid[:args.vid_len]
        vid = [vid[i] for i in range(0, len(vid), args.sample_period)]
        for j, frame in enumerate(vid):
            img = Image.open(frame)
            img = img.resize((args.img_size, args.img_size))
            img.save(f"{args.dest_path}/{name}/dangerous/{i}/{j}.jpg")
        pbar.update(1)
    pbar.close()
    

if __name__ == "__main__":
    args = parse_args()
    nusc_trainval = NuScenes(dataroot=args.data_path, version="v1.0-trainval", verbose=True)
    nusc_test = NuScenes(dataroot=args.data_path, version="v1.0-test", verbose=True)
    train_scenes = nusc_trainval.scene[:700]
    val_scenes = nusc_trainval.scene[700:]
    test_scenes = nusc_test.scene
    train_annotations = json.load(open(f"{args.ann_path}/train.json", "r"))
    val_annotations = json.load(open(f"{args.ann_path}/val.json", "r"))
    test_annotations = json.load(open(f"{args.ann_path}/test.json", "r"))
    train_video_paths = get_vids_paths(nusc_trainval, train_annotations, "train")
    val_video_paths = get_vids_paths(nusc_trainval, val_annotations, "val")
    test_video_paths = get_vids_paths(nusc_test, test_annotations, "test")
    Path(args.dest_path).mkdir(parents=True, exist_ok=True)
    make_dataset(args, train_video_paths, "train")
    make_dataset(args, val_video_paths, "val")
    make_dataset(args, test_video_paths, "test")
    print("Done.")