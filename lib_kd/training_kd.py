import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def train_kd_epoch(teacher, student, loader, optimizer, criterion,
                   scaler, device, temperature, alpha, beta):
    teacher.eval()
    student.train()
    total_sum, kd_sum, sup_sum = 0.0, 0.0, 0.0
    T = temperature

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            with torch.no_grad():
                t_logits = teacher(images).squeeze(1)
            s_logits = student(images).squeeze(1)

            t_soft = torch.sigmoid(t_logits / T).detach()
            kd_loss = F.binary_cross_entropy_with_logits(
                s_logits / T, t_soft, reduction="mean") * (T ** 2)
            sup_loss = criterion(s_logits, labels)
            total = alpha * kd_loss + beta * sup_loss

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        total_sum += total.item()
        kd_sum += kd_loss.item()
        sup_sum += sup_loss.item()

    n = len(loader)
    return total_sum / n, kd_sum / n, sup_sum / n


def compute_val_loss(student, loader, criterion, device):
    student.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = student(images).squeeze(1)
            total += criterion(out, labels).item()
    return total / len(loader)
