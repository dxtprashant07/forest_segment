import torch
import os
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, val_dice, history, path):
    """Saves model checkpoint with metadata"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'history': history
    }, path)
    print(f"  ✅ Checkpoint saved: {path}")

def plot_history(history, save_path=None):
    """Plots training history (Loss, Dice, IoU)"""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Dice
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    plt.plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    plt.title('Training and Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    # IoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    plt.plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  📊 History plot saved: {save_path}")
    
    plt.close()
