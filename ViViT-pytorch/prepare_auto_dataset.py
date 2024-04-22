from pathlib import Path
import argparse
import tqdm
from PIL import Image
import shutil
from nuscenes_edit import NuScenes
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--dest-path", type=str)
    parser.add_argument("--img-size", type=int, default=400)
    parser.add_argument("--vid-len", type=int, default=190)
    parser.add_argument("--sample-period", type=int, default=4)
    args = parser.parse_args()
    return args


def get_labels(scenes, keywords):
    labels = []
    for s in scenes:
        if (any(k in s["description"] for k in keywords)):
            labels.append(1)
        else:
            labels.append(0)
    return labels


def get_frame_paths(scenes, nusc, version):
    frame_paths = []
    pbar = tqdm.tqdm(total=len(scenes), position=0, leave=True, desc=f"Reading {version} data...")
    for s in scenes:
        paths = []
        scene_rec = nusc.get("scene", s["token"])
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["CAM_FRONT"])
        has_more_frames = True
        while has_more_frames:
            img_path, _, _ = nusc.get_sample_data(sd_rec["token"])
            paths.append(img_path)
            if not sd_rec["next"] == "":
                sd_rec = nusc.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        paths = paths[:args.vid_len]
        paths = [paths[i] for i in range(0, len(paths), args.sample_period)]
        frame_paths.append(paths)
        pbar.update(1)
    pbar.close()
    return frame_paths


def save_video(args, paths, label, vcount, version):
    for i, p in enumerate(paths):
        img = Image.open(p)
        img = img.resize((args.img_size, args.img_size))
        if label == 0:
            img_path = Path(f"{args.dest_path}/{version}/safe/{vcount['safe']}")
            img_path.mkdir(parents=True, exist_ok=True)
            img.save(img_path/f"{i}.jpg")
        elif label == 1:
            img_path = Path(f"{args.dest_path}/{version}/dangerous/{vcount['dangerous']}")
            img_path.mkdir(parents=True, exist_ok=True)
            img.save(img_path/f"{i}.jpg")
        else:
            raise(SystemError(f"label {label} not valid"))


def save_dataset(args, paths, labels, version):
    vcount = {"safe": 0, "dangerous": 0}
    pbar = tqdm.tqdm(total=len(paths), position=0, leave=True, desc=f"Saving {version} data...")
    for (p, l) in zip(paths, labels):
        save_video(args, p, l, vcount, version)
        if l == 0: 
            vcount["safe"] += 1
        elif l == 1:
            vcount["dangerous"] += 1
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    keywords = [
        "overtake",
        "overtaking",
        "ped crossing",
        "peds crossing",
        "cyclist crossing"
    ]
    nusc_trainval = NuScenes(dataroot=args.data_path, version="v1.0-trainval", verbose=True)
    nusc_test = NuScenes(dataroot=args.data_path, version="v1.0-test", verbose=True)
    train_scenes = nusc_trainval.scene[:700]
    val_scenes = nusc_trainval.scene[700:]
    train_labels = get_labels(train_scenes, keywords)
    val_labels = get_labels(val_scenes, keywords)
    train_paths = get_frame_paths(train_scenes, nusc_trainval, version = "train")
    val_paths = get_frame_paths(val_scenes, nusc_trainval, version = "val")
    if Path(args.dest_path).exists():
        shutil.rmtree(args.dest_path)
    save_dataset(args, train_paths, train_labels, version = "train")
    save_dataset(args, val_paths, val_labels, version = "val")

    