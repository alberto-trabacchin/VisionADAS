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
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-frames", type=int, default=190)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args


class NuScenesDS(Dataset):
    
    def __init__(self, args, version, transform, target_transform):
        self.root = Path(args.data_path)
        self.version = version
        if (version == "train") or (version == "val") or (version == "test"):
            self.data_path = Path(f"{args.data_path}/{version}")
        else:
            raise SystemError("Invalid version. Choose from 'train', 'val', 'test'.")
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.targets = self._load_data(self.data_path)

    def _load_data(self, data_path):
        safe_recs = Path(f"{data_path}/safe").iterdir()
        dang_recs = Path(f"{data_path}/dangerous").iterdir()
        data = []
        targets = []
        for rec in safe_recs:
            frames = [f for f in rec.iterdir()]
            data.append(frames)
            targets.append(0)
        for rec in dang_recs:
            frames = [f for f in rec.iterdir()]
            data.append(frames)
            targets.append(1)
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.targets[idx]
        images = []
        for path in self.data[idx]:
            img = Image.open(path)
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
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean = [0.4175, 0.4213, 0.4137], std = [0.2029, 0.2005, 0.2079])
    ])
    train_dataset = data.NuScenesDS(
        args,
        version = "train",
        transform=transforms,
        target_transform=None
    )
    val_dataset = data.NuScenesDS(
        args,
        version = "val",
        transform=transforms,
        target_transform=None
    )
    test_dataset = data.NuScenesDS(
        args,
        version = "test",
        transform=transforms,
        target_transform=None
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


def get_norm_params(args, train_loader):
    means_red = []
    means_green = []
    means_blue = []
    stds_red = []
    stds_green = []
    stds_blue = []
    for batch in train_loader:
        imgs, _ = batch
        imgs.to(args.device)
        means_red.append(imgs[:, :, [0]].mean().cpu())
        means_green.append(imgs[:, :, [1]].mean().cpu())
        means_blue.append(imgs[:, :, [2]].mean().cpu())
        stds_red.append(imgs[:, :, [0]].std().cpu())
        stds_green.append(imgs[:, :, [1]].std().cpu())
        stds_blue.append(imgs[:, :, [2]].std().cpu())
    means = torch.tensor([torch.stack(means_red).mean(), torch.stack(means_green).mean(), torch.stack(means_blue).mean()])
    stds = torch.tensor([torch.stack(stds_red).mean(), torch.stack(stds_green).mean(), torch.stack(stds_blue).mean()])
    return means, stds


if __name__ == "__main__":
    args = parse_args()
    train_dataset, val_dataset, test_dataset = get_nuscenes_data(args)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, 
        val_dataset, 
        test_dataset, 
        args
    )
    print(train_dataset[0][0])
    print(train_dataset[0][0].shape)
    mean, std = get_norm_params(args, train_loader)
    print(mean, std)
    