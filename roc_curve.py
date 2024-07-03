from pathlib import Path
import torch
from vit_pytorch import ViT
from collections import OrderedDict
import data
from torchvision.transforms import v2
import argparse
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--model-name', type=str, help='example: ViT-B32', required=True)
    parser.add_argument('--data-path', type=str, help='example: .../bdd100k')

    # Model params
    parser.add_argument('--img-size', type=int, default=512)

    # Training params
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def get_hyperparams(model_name):
    if model_name == "ViT-B32" or model_name == "ViT-B32-D941k":
        return {
            "img_size": 512,
            "patch_size": 32,
            "embed_dim": 768,
            "depth": 12,
            "heads": 12,
            "mlp_dim": 3072
        }
    elif model_name == "ViT-L32" or model_name == "ViT-L32-D941k":
        return {
            "img_size": 512,
            "patch_size": 32,
            "embed_dim": 1024,
            "depth": 24,
            "heads": 16,
            "mlp_dim": 4096
        }
    else:
        raise ValueError(f"Model name {model_name} not supported")
    

def load_model(model_name, model_path):
    # Get model hyperparameters based on the model name
    params = get_hyperparams(model_name)
    
    # Initialize the model with the obtained hyperparameters
    model = ViT(
        image_size=params["img_size"],
        patch_size=params["patch_size"],
        num_classes=2,  # Assuming binary classification
        dim=params["embed_dim"],
        depth=params["depth"],
        heads=params["heads"],
        mlp_dim=params["mlp_dim"]
    )
    mode = model_path.stem.split("-")[-1]
  
    # Load the model weights
    state_dict = torch.load(model_path)
    if mode == "MPL_student":
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def validate_model(model, val_loader):
    y_preds = []
    y_true = []
    ds_len = len(val_loader.dataset)
    pbar = tqdm(total=len(val_loader), desc="Validating", unit="batch")
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_preds.append(probs[:, 1].cpu().numpy())
            y_true.append(labels.cpu().numpy())
            pbar.update(1)
    pbar.close()
    y_preds = np.concatenate(y_preds)
    y_true = np.concatenate(y_true)
    return y_preds, y_true


def plot_roc_curve(y_true, y_preds_sl, y_preds_mpl):
    fpr_sl, tpr_sl, thresholds_sl = roc_curve(y_true, y_preds_sl)
    fpr_mpl, tpr_mpl, thresholds_mpl = roc_curve(y_true, y_preds_mpl)
    roc_auc_sl = auc(fpr_sl, tpr_sl)
    roc_auc_mpl = auc(fpr_mpl, tpr_mpl)

    plt.figure()
    lw = 2
    plt.plot(fpr_sl, tpr_sl, color='darkorange',
             lw=lw, label=f'SL ROC curve (area = {roc_auc_sl:.2f})')
    plt.plot(fpr_mpl, tpr_mpl, color='navy',
             lw=lw, label=f'MPL ROC curve (area = {roc_auc_mpl:.2f})')
        # Annotate specific thresholds
    for threshold in [0.2, 0.4, 0.5, 0.7]:
        # Find the closest threshold index in SL curve
        idx_sl = np.argmin(np.abs(thresholds_sl - threshold))
        plt.annotate(f'{threshold:.2f}', (fpr_sl[idx_sl], tpr_sl[idx_sl]), textcoords="offset points", xytext=(10,20), ha='center', color='darkorange')

        # Find the closest threshold index in MPL curve
        idx_mpl = np.argmin(np.abs(thresholds_mpl - threshold))
        plt.annotate(f'{threshold:.2f}', (fpr_mpl[idx_mpl], tpr_mpl[idx_mpl]), textcoords="offset points", xytext=(10,-20), ha='center', color='navy')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    sl_model_path = Path(f"./checkpoints/sl/{model_name}-SL.pth")
    mpl_model_path = Path(f"./checkpoints/mpl/{model_name}-MPL_student.pth")
    sl_roc_curve_path = Path(f"./roc_curve/{model_name}-SL.png")
    mpl_roc_curve_path = Path(f"./roc_curve/{model_name}-MPL.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = v2.Compose([
        v2.Resize((args.img_size, args.img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/val/',
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # Load the model
    sl_model = load_model(model_name, sl_model_path)
    mpl_model = load_model(model_name, mpl_model_path)

    model = sl_model
    model.to(device)
    model.eval()
    y_preds_sl, y_true = validate_model(model, val_loader)

    model = mpl_model
    model.to(device)
    model.eval()
    y_preds_mpl, y_true = validate_model(model, val_loader)
    np.save(f"auc_roc/{model_name}/y_true.npy", y_true)
    np.save(f"auc_roc/{model_name}/y_preds_sl.npy", y_preds_sl)
    np.save(f"auc_roc/{model_name}/y_preds_mpl.npy", y_preds_mpl)