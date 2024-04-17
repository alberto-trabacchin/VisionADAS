from torch.utils.data import Dataset
import nuscenes_edit
import json
from torchvision.io import read_image
from matplotlib import pyplot as plt
import torch
import tqdm
import data
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.data import DataLoader


class NuScenesDS(Dataset):
    def __init__(self, args, version, ann_path, n_frames, transform, target_transform=None, verbose=True):
        self.version = version
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        cache_path = "datasets_cache"
        Path(cache_path).mkdir(exist_ok=True)
        if not Path(f"{cache_path}/{version}.pth").exists():
            if (version == "train") or (version == "val"):
                self.nusc = nuscenes_edit.NuScenes(dataroot = args.data_path, version="v1.0-trainval", verbose=verbose)
            elif version == "test":
                self.nusc = nuscenes_edit.NuScenes(dataroot = args.data_path, version="v1.0-test", verbose=verbose)
            else:
                raise SystemError("Invalid version. Choose from 'train', 'val', 'test'.")
            
            if version == "train":
                self.scenes = self.nusc.scene[:700]
            elif version == "val":
                self.scenes = self.nusc.scene[700:]
            else:
                self.scenes = self.nusc.scene

            self.annotations = json.load(open(f"{ann_path}/{version}.json", "r"))
            self.data = []
            self.targets = []
            pbar = tqdm.tqdm(total=len(self.annotations), desc=f"Loading {version} data")
            for ann in self.annotations:
                scene_rec = self.nusc.get("scene", ann["token"])
                paths = self._get_frames_paths(scene_rec)
                self.data.append(paths)
                self.targets.append(ann["annotation_id"])
                pbar.update(1)
            pbar.close()
            torch.save({"data": self.data, "targets": self.targets}, f"datasets_cache/{version}.pth")
            print(f"Data saved to cache: datasets_cache/{version}.pth")
        else:
            print(f"Loading {version} data from cache...")
            cache = torch.load(f"{cache_path}/{version}.pth")
            self.data = cache["data"]
            self.targets = cache["targets"]
        
        self.data = [d[:n_frames] for d in self.data]


    def _get_frames_paths(self, scene_rec):
        sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", sample_rec["data"]["CAM_FRONT"])
        has_more_frames = True
        self.img_paths = []
        while has_more_frames:
            im_path, _, _ = self.nusc.get_sample_data(sd_rec["token"])
            self.img_paths.append(im_path)
            if not sd_rec["next"] == "":
                sd_rec = self.nusc.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        return self.img_paths

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target = self.targets[idx]
        images = []
        for path in self.data[idx]:
            img = read_image(path)
            img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        if self.target_transform:
            target = self.target_transform(target)
        if self.transform:
            images = self.transform(images)
        return images, target
    
    def get_weights(self):
        n_safe = self.targets.count(0)
        n_dang = self.targets.count(1)
        w_safe = n_safe / (n_safe + n_dang)
        w_dang = 1.0 - w_safe
        return torch.tensor([w_safe, w_dang])


def get_nuscenes_data(args):
    transforms = v2.Compose([
        v2.Resize(size=(args.img_size, args.img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = data.NuScenesDS(
        args, 
        version = "train", 
        ann_path=args.anns_path,
        n_frames=args.n_frames,
        transform=transforms,
        verbose = False
    )
    val_dataset = data.NuScenesDS(
        args, 
        version = "val", 
        ann_path=args.anns_path,
        n_frames=args.n_frames, 
        transform=transforms,
        verbose = False
    )
    test_dataset = data.NuScenesDS(
        args, 
        version = "test", 
        ann_path=args.anns_path,
        n_frames=args.n_frames,
        transform=transforms,
        verbose = False
    )
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset, args):
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True
    )
    return train_loader, val_loader, test_loader