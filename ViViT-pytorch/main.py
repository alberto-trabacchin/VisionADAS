import argparse
import data
from torch.utils.data import DataLoader
from pathlib import Path
from termcolor import colored
from vivit import ViViT
import wandb
import tqdm
import torch
import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ViViT")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--img-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=48)
    args = parser.parse_args()
    return args


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def file_debug(preds, labels, step, args):
    Path("./debug").mkdir(parents=True, exist_ok=True)
    Path("./debug/predictions.txt").touch(exist_ok=True)
    with open("./debug/predictions.txt", "a") as f:
        f.write(f"STEP: {step+1}/{args.train_steps}\n" \
                f"Predictions: {preds.argmax(dim=1)}\n" \
                f"Ground Truth: {labels}\n" \
                f"Accuracy: {metrics.accuracy(preds, labels):.4f}\n" \
                f"Precision: {metrics.precision(preds, labels):.4f}\n" \
                f"Recall: {metrics.recall(preds, labels):.4f}\n" \
                f"F1 Score: {metrics.f1_score(preds, labels):.4f}\n" \
                f"---------------------------------\n\n")


def train_loop(args, model, optimizer, criterion, train_loader, val_loader, scheduler):
    pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)
    train_lb_size = train_loader.dataset.__len__()
    val_size = val_loader.dataset.__len__()
    if Path("debug/predictions.txt").exists():
        Path("debug/predictions.txt").unlink()
    wandb.init(
        project='SceneUnderstanding',
        name=f'{args.name}_{train_lb_size}LB_{val_size}VL',
        config=args
    )
    train_iter = iter(train_loader)
    train_loss = metrics.AverageMeter()
    train_acc = metrics.AverageMeter()
    top1_acc = 0
    top_f1 = 0

    for step in range(args.train_steps):
        model.train()
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        imgs, labels = batch
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        train_loss.update(loss.item())
        train_acc.update(metrics.accuracy(preds, labels))
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.update(1)
        pbar.set_description(f"{step+1:4d}/{args.train_steps}  train/loss: {train_loss.avg :.4E} | train/acc: {train_acc.avg:.4f}")
        
        if (step + 1) % args.eval_steps == 0:
            pbar.close()
            pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
            model.eval()
            predictions = torch.tensor([], dtype=torch.float32)
            ground_truths = torch.tensor([], dtype=torch.int32)
            val_losses = torch.tensor([], dtype=torch.float32)
            for val_batch in val_loader:
                imgs, labels = val_batch
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                with torch.inference_mode():
                    preds = model(imgs)
                    predictions = torch.cat((predictions, preds.cpu()))
                    ground_truths = torch.cat((ground_truths, labels.int().cpu()))
                    loss = criterion(preds, labels)
                    val_losses = torch.cat((val_losses, torch.tensor([loss.item()])))
                pbar.update(1)

            val_loss = val_losses.mean().cpu()
            val_acc = metrics.accuracy(predictions, ground_truths)
            prec = metrics.precision(predictions, ground_truths)
            rec = metrics.recall(predictions, ground_truths)
            f1 = metrics.f1_score(predictions, ground_truths)
            pbar.set_description(f"{step+1:4d}/{args.train_steps}  VALID/loss: {val_loss:.4E} | VALID/acc: {val_acc:.4f}" \
                                 f" | VALID/prec: {prec:.4f} | VALID/rec: {rec:.4f} | VALID/f1: {f1:.4f}")
            pbar.close()
            if val_acc > top1_acc:
                top1_acc = val_acc
            if f1 > top_f1:
                top_f1 = f1
                save_path = Path('checkpoints/')
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f'{args.name}.pth'
                torch.save(model.state_dict(), save_path)
                wandb.save(f'{args.name}.pth')
                print(colored(f"--> Model saved at {save_path}", "yellow"))
            wandb.log({
                "train/loss": train_loss.avg,
                "train/acc": train_acc.avg,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "top1_acc": top1_acc,
                "top_f": top_f1
            }, step = step)
            # wandb.watch(models = model, log='all')
            print(f'top_f1: {top_f1:.6f}\n')
            train_loss.reset()
            train_acc.reset()
            file_debug(predictions, ground_truths, step, args)
            if (step + 1) != args.train_steps:
                pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)


if __name__ == "__main__":
    args = parse_args()
    train_dataset, val_dataset, test_dataset = data.get_nuscenes_data(args)
    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_dataset, 
        val_dataset, 
        test_dataset, 
        args
    )

    model = ViViT(
        image_size=args.img_size, 
        patch_size=20,
        num_classes=2,
        num_frames=args.n_frames,
        dim=300,
        depth=10,
        heads=4,
        in_channels=3,
        dim_head=128,
        dropout=0.1,
        emb_dropout=0.1,
        scale_dim=4
    ).to(args.device)

    criterion = torch.nn.CrossEntropyLoss(
        weight = train_dataset.get_weights().to(args.device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.train_steps, 
        eta_min=0.0001
    )

    train_loop(
        args, 
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        scheduler
    )
