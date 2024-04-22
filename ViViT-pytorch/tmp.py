from nuscenes_edit import NuScenes
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str)
    args = parser.parse_args()
    return args


def render_scene(scenes):
    for s in scenes:
        token = s["token"]
        nusc.render_scene_channel(
            token,
            "CAM_FRONT",
            imsize=(1280, 720),
            with_anns=False,
            wait_time=10
        )
        input(f"{s['description']}")


if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(dataroot=args.data_path, version="v1.0-trainval", verbose=True)
    keywords = [
        "overtake",
        "overtaking",
        "ped crossing",
        "peds crossing",
        "cyclist crossing"
    ]
    sel_scenes = [s for s in nusc.scene if any(k in s["description"] for k in keywords)]
    print(f"Found {len(sel_scenes)} scenes.")
    render_scene(sel_scenes)