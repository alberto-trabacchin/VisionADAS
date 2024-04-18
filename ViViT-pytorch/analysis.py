from matplotlib import pyplot as plt
import torch
import data
import metrics
import tqdm
import argparse
from vivit import ViViT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ViViT.pth")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--anns-path", type=str, default="./annotations/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=400)
    parser.add_argument("--n-frames", type=int, default=190)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args



def test_model(model, dataloader, criterion):
    pbar = tqdm.tqdm(total=len(dataloader), position=0, leave=True)
    val_loss = metrics.AverageMeter()
    val_acc = metrics.AverageMeter()
    predictions = torch.tensor([], dtype=torch.float32)
    ground_truths = torch.tensor([], dtype=torch.int32)
    for batch in dataloader:
        imgs, labels = batch
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        with torch.inference_mode():
            preds = model(imgs)
            predictions = torch.cat((predictions, preds.cpu()))
            ground_truths = torch.cat((ground_truths, labels.int().cpu()))
            loss = criterion(preds, labels)
            val_loss.update(loss.item())
            val_acc.update(metrics.accuracy(preds, labels))
        pbar.update(1)
    pbar.close()
    prec = metrics.precision(predictions, ground_truths)
    rec = metrics.recall(predictions, ground_truths)
    f1 = metrics.f1_score(predictions, ground_truths)
    return {"loss": val_loss.avg, "acc": val_acc.avg, "prec": prec, "rec": rec, "f1": f1}



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
        dim=192,
        depth=4,
        heads=4,
        in_channels=3,
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1,
        scale_dim=4
    ).to(args.device)
    model.load_state_dict(torch.load(f"./checkpoints/{args.model_name}"))
    
    criterion = torch.nn.CrossEntropyLoss()
    val_stats = test_model(model, val_loader, criterion)
    test_stats = test_model(model, test_loader, criterion)