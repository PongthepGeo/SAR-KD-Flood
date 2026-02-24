import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from lib.model import MLP_256
from lib.loss import FocalTverskyLoss
from lib.dataset import MultiChannelDataset
from lib.augmentation import get_training_augmentation, get_validation_augmentation

# ── Parameters ────────────────────────────────────────────────────────────────
SEED = 42
GPU_ID = 0
DATA_ROOT = "02_kfold"
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 40
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
WARMUP_EPOCHS = 3
FTL_ALPHA = 0.30
FTL_BETA = 0.70
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

train_ds = MultiChannelDataset(DATA_ROOT, split="train", transform=get_training_augmentation())
val_ds = MultiChannelDataset(DATA_ROOT, split="val", transform=get_validation_augmentation())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = MLP_256(in_channels=1, width=64, mlp_hidden=64, dropout=0.1).to(device)

criterion = FocalTverskyLoss(alpha=FTL_ALPHA, beta=FTL_BETA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
scaler = GradScaler('cuda')

best_val = float("inf")
wait = 0

for epoch in range(1, EPOCHS + 1):
    if epoch <= WARMUP_EPOCHS:
        warmup_lr = LR * epoch / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg["lr"] = warmup_lr

    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            out = model(images).squeeze(1)
            loss = criterion(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images).squeeze(1)
            val_loss += criterion(out, masks).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        marker = " *best*"
    else:
        wait += 1

    print(f"[{epoch:03d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}{marker}")

    if wait >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"Best val loss: {best_val:.4f}")
