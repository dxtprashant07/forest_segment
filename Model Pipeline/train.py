import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from src.data.dataset import ForestChangeDataset
from src.models.change_detection import SiameseUNet
from src.training.trainer import train_epoch, validate_epoch
from src.training.utils import save_checkpoint, plot_history

def main(args):
    print("\n" + "=" * 70)
    print("🚀 SATELLITE IMAGERY PIPELINE: MODEL TRAINING")
    print("=" * 70)
    
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ⚙️  Device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n📦 Loading Datasets...")
    # Example logic: Full training on specific AOIs, Partial on another
    # Adjust this logic based on actual folder structure or arguments
    # For now, we assume a standard split in the data_root
    
    train_dataset = ForestChangeDataset(args.data_root, split='train')
    val_dataset = ForestChangeDataset(args.data_root, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 2. Model Setup
    print("\n🏗️  Initializing Model...")
    model = SiameseUNet(in_channels=4, base_channels=args.base_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Training Loop
    best_val_dice = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_dice': [], 'train_iou': [], 
               'val_loss': [], 'val_dice': [], 'val_iou': []}
    
    print("\n🔥 Starting Training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train & Val
        t_loss, t_dice, t_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_dice, v_iou = validate_epoch(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(v_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update History
        history['train_loss'].append(t_loss)
        history['train_dice'].append(t_dice)
        history['train_iou'].append(t_iou)
        history['val_loss'].append(v_loss)
        history['val_dice'].append(v_dice)
        history['val_iou'].append(v_iou)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch}/{args.epochs}] - {epoch_time:.0f}s | "
              f"Train: Loss={t_loss:.4f} Dice={t_dice:.4f} | "
              f"Val: Loss={v_loss:.4f} Dice={v_dice:.4f} | LR={current_lr:.1e}")
        
        # Checkpoint
        if v_dice > best_val_dice + 0.005:  # Min delta
            best_val_dice = v_dice
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, v_dice, history, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break
                
    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time/60:.1f} minutes")
    
    # Plot history
    plot_history(history, save_path=os.path.join(args.results_dir, 'training_history.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    main(args)
