import argparse
import nuscenes_edit
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--anns-path", type=str, default="./annotations/")
    parser.add_argument("--version", type=str, default="train")
    parser.add_argument("--img-size", nargs="+", type=int, default=[1280, 720])
    parser.add_argument("--get-info", action="store_true")
    args = parser.parse_args()
    args.img_size = tuple(args.img_size)
    args.anns_fpath = Path(args.anns_path) / f"{args.version}.json"
    return args


def get_data(args):
    if args.version == "train":
        nusc = nuscenes_edit.NuScenes(dataroot = args.data_path, version='v1.0-trainval', verbose=True)
        scenes = nusc.scene[:700]
    elif args.version == "val":
        nusc = nuscenes_edit.NuScenes(dataroot = args.data_path, version='v1.0-trainval', verbose=True)
        scenes = nusc.scene[700:]
    elif args.version == "test":
        nusc = nuscenes_edit.NuScenes(dataroot = args.data_path, version='v1.0-test', verbose=True)
        scenes = nusc.scene
    else:
        print(SystemError("Invalid version. Choose from 'train', 'val', 'test'."))
        exit()
    return nusc, scenes


def get_annotations(args):
    Path(args.anns_path).mkdir(parents=True, exist_ok=True)
    if args.anns_fpath.exists():
        annotations = json.load(open(args.anns_fpath, "r"))
    else:
        annotations = []
    return annotations


def render_scene(anns, nusc, scenes, args):
    ann_tokens = [t["token"] for t in anns]
    new_scenes = [t for t in scenes if t["token"] not in ann_tokens]
    if not new_scenes:
        print("All scenes have been rendered.")
        exit(0)
    else:
        scene = new_scenes[0]
        target = 3
        while target == 3:
            token = scene["token"]
            nusc.render_scene_channel(
                token, 
                'CAM_FRONT', 
                imsize=args.img_size,
                with_anns = False,
                wait_time = 10
            )
            target = int(input(f"Annotation [1:Safe / 2:Dangerous / 3:Repeat / 0:Exit] --> "))
        if target == 0:
            exit(0)
        id2ann = {
            1: "Safe",
            2: "Dangerous",
        }
        scene["annotation"] = id2ann[target]
        scene["annotation_id"] = int(target) - 1
    return scene


def get_anns_info(args):
    anns = json.load(open(args.anns_fpath, "r"))
    anns_info = {
        "Safe": 0,
        "Dangerous": 0,
    }
    for ann in anns:
        anns_info[ann["annotation"]] += 1
    return anns_info


if __name__ == "__main__":
    args = parse_args()
    if args.get_info:
        anns_info = get_anns_info(args)
        print(anns_info)
        exit(0)
    anns = get_annotations(args)
    nusc, scenes = get_data(args)
    while True:
        ann_scene = render_scene(anns, nusc, scenes, args)
        anns.append(ann_scene)
        json.dump(anns, open(args.anns_fpath, "w"), indent=4)
