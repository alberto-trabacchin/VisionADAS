import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter


class CustomBDD100kDataset(Dataset):
    def __init__(self, root_dir, transform=None, labeled=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
            labeled (bool, optional): Indicates if the dataset is labeled.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled
        self.samples = []

        if labeled:
            for label_dir in ['class_0', 'class_1']:
                class_label = int(label_dir.split('_')[-1])
                class_dir = os.path.join(root_dir, label_dir)
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_file), class_label))
        else:
            for img_file in os.listdir(root_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(root_dir, img_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.labeled:
            img_path, label = self.samples[idx]
        else:
            img_path = self.samples[idx]
            label = -1  # Use -1 or another placeholder for unlabeled data

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
    
    def get_weights(self):
        counts = Counter([s[1] for s in self.samples])
        weights = [1.0 / c for c in counts.values()]
        return weights




# Define DataLoader Functions
def create_data_loaders(train_dataset, val_dataset, args):
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    # Define Transformations
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # Resize to the same size as the preprocessed images
        transforms.ToTensor(),  # Convert images to tensors
    ])

    # Instantiate Dataset Objects for Train and Validation Sets
    train_dataset = CustomBDD100kDataset(
        root_dir='/home/alberto-trabacchin-wj/datasets/bdd100k/custom_dataset/train/',
        transform=transform
    )

    val_dataset = CustomBDD100kDataset(
        root_dir='/home/alberto-trabacchin-wj/datasets/bdd100k/custom_dataset/val/',
        transform=transform
    )
    # Create Data Loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)