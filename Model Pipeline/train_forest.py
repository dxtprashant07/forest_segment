"""
Forest Segmentation Training Script
Trains mU-Net model for pixel-level forest classification.

Usage:
    python train_forest.py --data_root dataset --epochs 150 --batch_size 8
"""

import argparse
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from src.models.forest_segmentation import mUnet_model


def lr_decay(epoch):
    """Decays learning rate by 0.5 every 7 epochs"""
    init_alpha = 0.001
    factor = 0.5
    drop_every = 7
    alpha = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))
    return float(alpha)


def load_dataset(data_root):
    """Loads NumPy patches from dataset directory."""
    image_dir = os.path.join(data_root, 'images_npy')
    mask_dir = os.path.join(data_root, 'masks_npy')
    
    print("="*70)
    print("📂 LOADING TRAINING DATA")
    print("="*70)
    
    image_dataset = []
    mask_dataset = []
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
    print(f"   Found {len(images)} image files")
    
    for i, image_name in enumerate(images):
        image = np.load(os.path.join(image_dir, image_name))
        image_dataset.append(image)
        
        if (i + 1) % 500 == 0:
            print(f"   Loaded {i+1}/{len(images)} images...")
    
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
    print(f"   Found {len(masks)} mask files")
    
    for i, mask_name in enumerate(masks):
        mask = np.load(os.path.join(mask_dir, mask_name))
        mask_dataset.append(mask)
    
    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)
    
    if len(mask_dataset.shape) == 3:
        mask_dataset = np.expand_dims(mask_dataset, axis=3)
    
    print(f"\n✅ Data Loaded:")
    print(f"   Images: {image_dataset.shape}")
    print(f"   Masks:  {mask_dataset.shape}")
    
    return image_dataset, mask_dataset


def main():
    parser = argparse.ArgumentParser(description="Train mU-Net for Forest Segmentation")
    parser.add_argument("--data_root", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", default="output", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    image_dataset, mask_dataset = load_dataset(args.data_root)
    
    # Split data (80/10/10)
    print("\n" + "="*70)
    print("🔀 SPLITTING DATA")
    print("="*70)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_dataset, mask_dataset,
        test_size=0.20,
        random_state=args.random_state
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=args.random_state
    )
    
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Val:   {X_val.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    
    # Build model
    print("\n" + "="*70)
    print("🏗️  BUILDING mU-NET MODEL")
    print("="*70)
    
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    model = mUnet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    print(f"   ✅ Model built: {model.count_params():,} parameters")
    
    # Callbacks
    print("\n" + "="*70)
    print("⚙️  CONFIGURING CALLBACKS")
    print("="*70)
    
    lr_scheduler = LearningRateScheduler(lr_decay, verbose=0)
    print("   ✅ LR Scheduler: 0.001 → decay by 0.5 every 7 epochs")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    print("   ✅ Early Stopping: patience=20")
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    print(f"   ✅ Checkpoints: {checkpoint_dir}")
    
    callbacks = [lr_scheduler, early_stop, checkpoint]
    
    # Train
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'Forest_Segmentation_Best.keras')
    model.save(final_path)
    print(f"\n✅ Final model saved: {final_path}")
    
    # Save history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"   History saved: {history_path}")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("📊 EVALUATING ON TEST SET")
    print("="*70)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
