import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from lib.loss import FocalTverskyLoss
from lib.dataset import MultiChannelDataset
from lib.augmentation import get_training_augmentation, get_validation_augmentation
from lib_kd.model_kd import create_teacher_model, create_student_model
from lib_kd.training_kd import train_kd_epoch, compute_val_loss

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

# KD parameters
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.5
KD_BETA = 0.5

# Teacher: PSPNet with MobileNetV2
TEACHER_TYPE = "PSPNet"
TEACHER_ENCODER = "mobilenet_v2"
TEACHER_ENCODER_WEIGHTS = "imagenet"

# Student: PSPMixer
PSPMIXER_PATCH = 32
PSPMIXER_HIDDEN = 128
PSPMIXER_DEPTH = 3
PSPMIXER_TOKENS_MLP = 64
PSPMIXER_CHANNELS_MLP = 256
# ──────────────────────────────────────────────────────────────────────────────

print("WARNING: use pretrained weight is imagenet")
print("enhance model performance need to request pretrained weight")

torch.manual_seed(SEED)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

train_ds = MultiChannelDataset(DATA_ROOT, split="train", transform=get_training_augmentation())
val_ds = MultiChannelDataset(DATA_ROOT, split="val", transform=get_validation_augmentation())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Teacher (frozen)
teacher = create_teacher_model(
    model_type=TEACHER_TYPE, encoder=TEACHER_ENCODER,
    encoder_weights=TEACHER_ENCODER_WEIGHTS, in_channels=1, classes=1
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# Student (trainable)
student = create_student_model(
    in_channels=1, num_classes=1,
    patch=PSPMIXER_PATCH, hidden=PSPMIXER_HIDDEN, depth=PSPMIXER_DEPTH,
    tokens_mlp=PSPMIXER_TOKENS_MLP, channels_mlp=PSPMIXER_CHANNELS_MLP
).to(device)

criterion = FocalTverskyLoss(alpha=FTL_ALPHA, beta=FTL_BETA)
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
scaler = GradScaler()

best_val = float("inf")
wait = 0

for epoch in range(1, EPOCHS + 1):
    if epoch <= WARMUP_EPOCHS:
        warmup_lr = LR * epoch / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg["lr"] = warmup_lr

    total_loss, kd_loss, sup_loss = train_kd_epoch(
        teacher, student, train_loader, optimizer, criterion,
        scaler, device, KD_TEMPERATURE, KD_ALPHA, KD_BETA
    )

    val_loss = compute_val_loss(student, val_loader, criterion, device)
    scheduler.step(val_loss)

    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        marker = " *best*"
    else:
        wait += 1

    print(f"[{epoch:03d}/{EPOCHS}] total={total_loss:.4f}  kd={kd_loss:.4f}  "
          f"sup={sup_loss:.4f}  val={val_loss:.4f}{marker}")

    if wait >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"Best val loss: {best_val:.4f}")
