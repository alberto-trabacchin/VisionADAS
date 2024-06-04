import argparse
import torch
import tqdm
import wandb
from pathlib import Path
from termcolor import colored
import numpy as np
import random
from torch.utils.data import DataLoader
import data
from vit_pytorch import ViT
import numpy as np
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--data-path', type=str, help='example: .../bdd100k')

    # Model params
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--mlp-dim', type=int, default=2048)

    # Training params
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eta-min', type=float, default=1e-5)
    args = parser.parse_args()
    return args


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_loop(args, model, optimizer, criterion, train_loader, val_loader, scheduler):
    # Metrics
    train_loss = AverageMeter()
    val_loss = AverageMeter()
    train_acc = AverageMeter()
    val_acc = AverageMeter()
    train_prec = AverageMeter()
    val_prec = AverageMeter()
    train_rec = AverageMeter()
    val_rec = AverageMeter()
    train_f1 = AverageMeter()
    val_f1 = AverageMeter()
    top_val_acc = 0.0
    top_val_f1 = 0.0
    min_val_loss = np.inf

    for epoch in range(args.epochs):
        # Training
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), position=0, leave=True, desc="Training...")

        for batch in train_loader:
            imgs, labels = batch
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            train_loss.update(loss.item())
            preds = preds.argmax(dim=1).cpu()
            labels = labels.cpu()
            train_acc.update(accuracy_score(preds, labels))
            train_prec.update(precision_score(preds, labels, zero_division=0))
            train_rec.update(recall_score(preds, labels, zero_division=0))
            train_f1.update(f1_score(preds, labels, zero_division=0))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.update(1)
        
        pbar.set_description(
            f"{epoch+1:4d}/{args.epochs} " \
            f"train/loss: {train_loss.avg:.4E} | " \
            f"train/acc: {train_acc.avg:.4f} | " \
            f"train/prec: {train_prec.avg:.4f} | " \
            f"train/rec: {train_rec.avg:.4f} | " \
            f"train/f1: {train_f1.avg:.4f}"
        )
        pbar.close()

        # Validation
        pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
        model.eval()
        for val_batch in val_loader:
            imgs, labels = val_batch
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            with torch.inference_mode():
                preds = model(imgs)
                loss = criterion(preds, labels)
                val_loss.update(loss.item())
                preds = preds.argmax(dim=1).cpu()
                labels = labels.cpu()
                val_acc.update(accuracy_score(preds, labels))
                val_prec.update(precision_score(preds, labels, zero_division=0))
                val_rec.update(recall_score(preds, labels, zero_division=0))
                val_f1.update(f1_score(preds, labels, zero_division=0))
            pbar.update(1)
        pbar.set_description(
            f"{epoch+1:4d}/{args.epochs} "
            f"VALID/loss: {val_loss.avg:.4E} | "
            f"VALID/acc: {val_acc.avg:.4f} | "
            f"VALID/prec: {val_prec.avg:.4f} | "
            f"VALID/rec: {val_rec.avg:.4f} | "
            f"VALID/f1: {val_f1.avg:.4f}"
        )
        pbar.close()

        # Save model
        if val_loss.avg < min_val_loss:
            min_val_loss = val_loss.avg
            save_path = Path('checkpoints/sl/')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f'{args.name}.pth'
            torch.save(model.state_dict(), save_path)
            wandb.save(f'{args.name}.pth')
            print(colored(f"--> Model saved at {save_path}", "yellow"))

        if val_acc.avg > top_val_acc:
            top_val_acc = val_acc.avg

        if val_f1.avg > top_val_f1:
            top_val_f1 = val_f1.avg

        # Wandb logging
        wandb.log({
            "train/loss": train_loss.avg,
            "train/acc": train_acc.avg,
            "train/prec": train_prec.avg,
            "train/rec": train_rec.avg,
            "train/f1": train_f1.avg,
            "val/loss": val_loss.avg,
            "val/acc": val_acc.avg,
            "val/prec": val_prec.avg,
            "val/rec": val_rec.avg,
            "val/f1": val_f1.avg,
            "top_val_acc": top_val_acc,
            "top_val_f1": top_val_f1,
            "min_val_loss": min_val_loss,
        }, step = epoch + 1)
        print(f'top_f1: {top_val_f1:.6f}\n' \
              f'top_acc: {top_val_acc:.6f}\n' \
              f'min_loss: {min_val_loss:.6f} \n')

        # Reset metrics
        val_loss.reset()
        val_acc.reset()
        train_loss.reset()
        train_acc.reset()
        train_prec.reset()
        val_prec.reset()
        train_rec.reset()
        val_rec.reset()
        train_f1.reset()
        val_f1.reset()


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = parse_args()
    set_seeds(args.seed)
    wandb.init(
        project='VisionADAS',
        name=f'{args.name}',
        config=args
    )
    
    transform = v2.Compose([
        v2.Resize((args.img_size, args.img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/train/',
        transform=transform
    )
    val_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/val/',
        transform=transform
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers
    )

    model = ViT( 
        image_size = args.img_size,
        patch_size = args.patch_size,
        num_classes = 2,
        dim = args.embed_dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim
    )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_weights = train_dataset.get_weights()
    criterion = torch.nn.CrossEntropyLoss(
        weight = torch.tensor(data_weights).to(args.device)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.eta_min
    )

    train_loop(
        args = args,
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        train_loader = train_loader,
        val_loader = val_loader,
        scheduler = scheduler
    )