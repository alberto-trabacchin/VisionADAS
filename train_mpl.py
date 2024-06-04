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
import torch.nn.functional as F


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


def train_loop(args, teacher, student,
               teacher_optim, student_optim,
               teacher_sched, student_sched,
               train_lb_loader, train_ul_loader, val_loader):
    # Metrics
    teach_train_loss = AverageMeter()
    teach_train_acc = AverageMeter()
    teach_train_prec = AverageMeter()
    teach_train_rec = AverageMeter()
    teach_train_f1 = AverageMeter()

    teach_val_loss = AverageMeter()
    teach_val_acc = AverageMeter()
    teach_val_prec = AverageMeter()
    teach_val_rec = AverageMeter()
    teach_val_f1 = AverageMeter()
    teach_top_val_acc = 0.0
    teach_top_val_f1 = 0.0
    teach_min_val_loss = np.inf

    stud_train_loss = AverageMeter()
    stud_train_acc = AverageMeter()
    stud_train_prec = AverageMeter()
    stud_train_rec = AverageMeter()
    stud_train_f1 = AverageMeter()

    stud_val_loss = AverageMeter()
    stud_val_acc = AverageMeter()
    stud_val_prec = AverageMeter()
    stud_val_rec = AverageMeter()
    stud_val_f1 = AverageMeter()
    stud_top_val_acc = 0.0
    stud_top_val_f1 = 0.0
    stud_min_val_loss = np.inf

    for epoch in range(args.epochs):
        count = 0
        print(f"\n{'-' * 80}")
        teacher.train()
        student.train()
        pbar = tqdm.tqdm(total=len(train_lb_loader), position=0, leave=True, desc=f"{epoch+1:4d}/{args.epochs}")
        for batch_lb in train_lb_loader:
            try:
                batch_ul = next(train_ul_iter)
            except:
                train_ul_iter = iter(train_ul_loader)
                batch_ul = next(train_ul_iter)
        
            imgs_lb, targets = batch_lb
            imgs_ul, _ = batch_ul
            imgs_lb, targets = imgs_lb.to(args.device), targets.to(args.device)
            imgs_ul = imgs_ul.to(args.device)
            teacher_optim.zero_grad()
            student_optim.zero_grad()

            teacher_lb_logits = teacher(imgs_lb)
            teacher_ul_logits = teacher(imgs_ul)
            teacher_ul_targets = torch.softmax(teacher_ul_logits, dim=1)

            student_lb_logits = student(imgs_lb)
            student_ul_logits = student(imgs_ul)

            # train the student
            student_optim.zero_grad()
            student_loss = F.cross_entropy(student_lb_logits, targets, reduction="mean")
            student_loss.backward()
            student_grad_1 = [p.grad.data.clone().detach() for p in student.parameters()]

            # train the student
            student_optim.zero_grad()
            student_loss = F.cross_entropy(student_ul_logits, teacher_ul_targets.detach(), reduction="mean")
            student_loss.backward()
            student_grad_2 = [p.grad.data.clone().detach() for p in student.parameters()]
            student_optim.step()
            student_sched.step()

            mpl_coeff = sum([torch.dot(g_1.ravel(), g_2.ravel()).sum().detach().item() for g_1, g_2 in zip(student_grad_1, student_grad_2)])

            # train the teacher
            teacher_optim.zero_grad()
            teacher_loss_ent = F.cross_entropy(teacher_lb_logits, targets, reduction="mean")
            teacher_loss_mpl = mpl_coeff * F.cross_entropy(teacher_ul_logits, teacher_ul_targets.detach(), reduction="mean")

            teacher_loss = teacher_loss_ent + teacher_loss_mpl

            teacher_loss.backward()
            teacher_optim.step()
            teacher_sched.step()

            teach_preds = teacher_lb_logits.argmax(dim=1).cpu()
            stud_preds = student_lb_logits.argmax(dim=1).cpu()
            targets = targets.cpu()

            teach_train_loss.update(teacher_loss.item())
            teach_train_acc.update(accuracy_score(teach_preds, targets))
            teach_train_prec.update(precision_score(teach_preds, targets, zero_division=0))
            teach_train_rec.update(recall_score(teach_preds, targets, zero_division=0))
            teach_train_f1.update(f1_score(teach_preds, targets, zero_division=0))

            stud_train_loss.update(student_loss.item())
            stud_train_acc.update(accuracy_score(stud_preds, targets))
            stud_train_prec.update(precision_score(stud_preds, targets, zero_division=0))
            stud_train_rec.update(recall_score(stud_preds, targets, zero_division=0))
            stud_train_f1.update(f1_score(stud_preds, targets, zero_division=0))

            pbar.update(1)

            count += 1
            if count == 10:
                break
        
        pbar.set_description(
            f"T: {epoch+1:4d}/{args.epochs} "
            f"t/loss: {teach_train_loss.avg:.4E} | "
            f"t/acc: {teach_train_acc.avg:.4f} | "
            f"t/prec: {teach_train_prec.avg:.4f} | "
            f"t/rec: {teach_train_rec.avg:.4f} | "
            f"t/f1: {teach_train_f1.avg:.4f}"
        )
        pbar.close()
        print(
            f"{' ' * 12}" \
            f"s/loss: {stud_train_loss.avg:.4E} | "
            f"s/acc: {stud_train_acc.avg:.4f} | "
            f"s/prec: {stud_train_prec.avg:.4f} | "
            f"s/rec: {stud_train_rec.avg:.4f} | "
            f"s/f1: {stud_train_f1.avg:.4f}\n"
        )

        # Validation
        pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
        with torch.inference_mode():
            teacher.eval()
            student.eval()
            for batch in val_loader:
                imgs, labels = batch
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                teach_logits = teacher(imgs)
                stud_logits = student(imgs)
                teacher_loss = F.cross_entropy(teach_logits, labels, reduction="mean")
                student_loss = F.cross_entropy(stud_logits, labels, reduction="mean")
                
                teach_preds = teach_logits.argmax(dim=1).cpu()
                stud_preds = stud_logits.argmax(dim=1).cpu()
                labels = labels.cpu()

                teach_val_loss.update(teacher_loss.item())
                teach_val_acc.update(accuracy_score(teach_preds, labels))
                teach_val_prec.update(precision_score(teach_preds, labels, zero_division=0))
                teach_val_rec.update(recall_score(teach_preds, labels, zero_division=0))
                teach_val_f1.update(f1_score(teach_preds, labels, zero_division=0))

                stud_val_loss.update(student_loss.item())
                stud_val_acc.update(accuracy_score(stud_preds, labels))
                stud_val_prec.update(precision_score(stud_preds, labels, zero_division=0))
                stud_val_rec.update(recall_score(stud_preds, labels, zero_division=0))
                stud_val_f1.update(f1_score(stud_preds, labels, zero_division=0))

                pbar.update(1)

        pbar.set_description(
            f"V: {epoch+1:4d}/{args.epochs} "
            f"t/loss: {teach_val_loss.avg:.4E} | "
            f"t/acc: {teach_val_acc.avg:.4f} | "
            f"t/prec: {teach_val_prec.avg:.4f} | "
            f"t/rec: {teach_val_rec.avg:.4f} | "
            f"t/f1: {teach_val_f1.avg:.4f}"
        )
        pbar.close()
        print(
            f"{' ' * 12}" \
            f"s/loss: {stud_val_loss.avg:.4E} | "
            f"s/acc: {stud_val_acc.avg:.4f} | "
            f"s/prec: {stud_val_prec.avg:.4f} | "
            f"s/rec: {stud_val_rec.avg:.4f} | "
            f"s/f1: {stud_val_f1.avg:.4f}"
        )

        # Save model
        if teach_val_loss.avg < teach_min_val_loss:
            teach_min_val_loss = teach_val_loss.avg
            save_path = Path('checkpoints/mpl/')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f'{args.name}_teacher.pth'
            torch.save(teacher.state_dict(), str(save_path))
            wandb.save(f'{args.name}_teacher.pth')
            print(colored(f"--> Model saved at {save_path}", "blue"))
        if stud_val_loss.avg < stud_min_val_loss:
            stud_min_val_loss = stud_val_loss.avg
            save_path = Path('checkpoints/mpl/')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f'{args.name}_student.pth'
            torch.save(student.state_dict(), str(save_path))
            wandb.save(f'{args.name}_student.pth')
            print(colored(f"--> Model saved at {save_path}", "yellow"))
        
        if stud_val_f1.avg > stud_top_val_f1:
            stud_top_val_f1 = stud_val_f1.avg
        if teach_val_f1.avg > teach_top_val_f1:
            teach_top_val_f1 = teach_val_f1.avg
        
        if stud_val_acc.avg > stud_top_val_acc:
            stud_top_val_acc = stud_val_acc.avg
        if teach_val_acc.avg > teach_top_val_acc:
            teach_top_val_acc = teach_val_acc.avg

        print(
            f"\n"\
            f"t/j: {teach_top_val_f1:.4E} | " \
            f"t/top-acc: {teach_top_val_acc:.4E} | " \
            f"t/min-loss: {teach_min_val_loss:.4E} \n" \
            f"s/top-f1: {stud_top_val_f1:.4E} | " \
            f"s/top-acc: {stud_top_val_acc:.4E} | " \
            f"s/min-loss: {stud_min_val_loss:.4E}" \
        )

        # Wandb logging
        wandb.log({
            "teach/train/loss": teach_train_loss.avg,
            "teach/train/acc": teach_train_acc.avg,
            "teach/train/prec": teach_train_prec.avg,
            "teach/train/rec": teach_train_rec.avg,
            "teach/train/f1": teach_train_f1.avg,
            "teach/val/loss": teach_val_loss.avg,
            "teach/val/acc": teach_val_acc.avg,
            "teach/val/prec": teach_val_prec.avg,
            "teach/val/rec": teach_val_rec.avg,
            "teach/val/f1": teach_val_f1.avg,
            "teach/top_val_acc": teach_top_val_acc,
            "teach/top_val_f1": teach_top_val_f1,
            "teach/min_val_loss": teach_min_val_loss,
            "stud/train/loss": stud_train_loss.avg,
            "stud/train/acc": stud_train_acc.avg,
            "stud/train/prec": stud_train_prec.avg,
            "stud/train/rec": stud_train_rec.avg,
            "stud/train/f1": stud_train_f1.avg,
            "stud/val/loss": stud_val_loss.avg,
            "stud/val/acc": stud_val_acc.avg,
            "stud/val/prec": stud_val_prec.avg,
            "stud/val/rec": stud_val_rec.avg,
            "stud/val/f1": stud_val_f1.avg,
            "stud/top_val_acc": stud_top_val_acc,
            "stud/top_val_f1": stud_top_val_f1,
            "stud/min_val_loss": stud_min_val_loss,
        }, step = epoch + 1)

        # Reset metrics
        teach_val_loss.reset()
        teach_val_acc.reset()
        teach_val_prec.reset()
        teach_val_rec.reset()
        teach_val_f1.reset()
        teach_train_loss.reset()
        teach_train_acc.reset()
        teach_train_prec.reset()
        teach_train_rec.reset()
        teach_train_f1.reset()

        stud_val_loss.reset()
        stud_val_acc.reset()
        stud_val_prec.reset()
        stud_val_rec.reset()
        stud_val_f1.reset()
        stud_train_loss.reset()
        stud_train_acc.reset()
        stud_train_prec.reset()
        stud_train_rec.reset()
        stud_train_f1.reset()


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

    # Transforms
    transform = v2.Compose([
        v2.Resize((args.img_size, args.img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_lb_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/train/',
        transform=transform,
        labeled=True
    )
    train_ul_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/unlabeled/',
        transform=transform,
        labeled=False
    )
    val_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/val/',
        transform=transform
    )

    # Dataloaders
    train_lb_loader = DataLoader(
        dataset = train_lb_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers
    )
    train_ul_loader = DataLoader(
        dataset = train_ul_dataset,
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

    # Models
    teacher = ViT( 
        image_size = args.img_size,
        patch_size = args.patch_size,
        num_classes = 2,
        dim = args.embed_dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim
    )
    student = ViT(
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
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)
    teacher.to(args.device)
    student.to(args.device)

    teacher_optim = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    student_optim = torch.optim.Adam(student.parameters(), lr=args.lr)

    teacher_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_optim, 
        T_max=args.epochs, 
        eta_min=args.eta_min
    )
    student_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optim, 
        T_max=args.epochs, 
        eta_min=args.eta_min
    )

    train_loop(
        args = args,
        teacher = teacher,
        student = student,
        teacher_optim = teacher_optim,
        student_optim = student_optim,
        teacher_sched = teacher_sched,
        student_sched = student_sched,
        train_lb_loader = train_lb_loader,
        train_ul_loader = train_ul_loader,
        val_loader = val_loader
    )