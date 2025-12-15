import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {DEVICE}")

TRAIN_IMAGE_PATH = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/train'
VAL_IMAGE_PATH   = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/val'
TRAIN_MASK_PATH  = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/train_mask'
VAL_MASK_PATH    = '/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS/val_mask'

NUM_CLASSES = 3  #
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30     
LEARNING_RATE = 2e-4

class BCSSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load Image
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_path = os.path.join(self.mask_dir, img_name)
        mask = np.array(Image.open(mask_path)) 

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
            
        return image, mask.long()

transforms_train = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
    ], p=0.2),
    A.CoarseDropout(max_holes=8, max_height=IMG_SIZE//10, max_width=IMG_SIZE//10, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

train_ds = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transforms_train)
val_ds   = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transforms_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", 
    num_labels=NUM_CLASSES, 
    ignore_mismatched_sizes=True
).to(DEVICE)

weights = torch.tensor([0.2, 0.5, 0.8]).to(DEVICE)

def criterion(logits, targets):
    return F.cross_entropy(logits, targets, weight=weights, label_smoothing=0.05)

def compute_iou(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan')) 
        else:
            ious.append(intersection / union)
    return np.nanmean(ious) 

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_loader)*2, num_training_steps=len(train_loader)*EPOCHS
)

best_iou = 0.0
print("Training...")

for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    train_loss, train_iou = 0.0, 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        train_iou += compute_iou(preds, masks, NUM_CLASSES)
        
    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou  = train_iou / len(train_loader)

    model.eval()
    val_loss, val_iou = 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(pixel_values=images)
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            loss = criterion(logits, masks)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_iou += compute_iou(preds, masks, NUM_CLASSES)
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou  = val_iou / len(val_loader)
    
    print(f"DONE Epoch {epoch} | Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if avg_val_iou > best_iou:
        print(f" New Best IoU: {avg_val_iou:.4f} (was {best_iou:.4f}) --> Saving...")
        best_iou = avg_val_iou
        torch.save(model.state_dict(), "best_segformer_clean.pth")

print(f"\nFinish! Best mIoU: {best_iou:.4f}")
