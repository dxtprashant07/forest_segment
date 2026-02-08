import torch
from src.training.metrics import dice_score, iou_score

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Training loop for one epoch (batch-wise metrics)"""
    model.train()

    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    num_batches = 0

    for t1, t2, mask, aoi_names in dataloader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device).unsqueeze(1)

        logits = model(t1, t2)
        loss = criterion(logits, mask)

        if torch.isnan(loss):
            print("❌ ERROR: NaN loss detected.")
            raise ValueError("NaN loss encountered")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred = torch.sigmoid(logits) > 0.5
            pred = pred.float()

            batch_dice = dice_score(pred, mask)
            batch_iou = iou_score(pred, mask)

        epoch_loss += loss.item()
        epoch_dice += batch_dice
        epoch_iou += batch_iou
        num_batches += 1

    return epoch_loss / num_batches, epoch_dice / num_batches, epoch_iou / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validation loop for one epoch (batch-wise metrics)"""
    model.eval()

    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for t1, t2, mask, aoi_names in dataloader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device).unsqueeze(1)

            logits = model(t1, t2)
            loss = criterion(logits, mask)

            pred = torch.sigmoid(logits) > 0.5
            pred = pred.float()

            batch_dice = dice_score(pred, mask)
            batch_iou = iou_score(pred, mask)

            epoch_loss += loss.item()
            epoch_dice += batch_dice
            epoch_iou += batch_iou
            num_batches += 1

    return epoch_loss / num_batches, epoch_dice / num_batches, epoch_iou / num_batches
