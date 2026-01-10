import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast # Added for AMP
from tqdm import tqdm
from metrics import dice_loss
from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CoronaryArterySegmentationDataset, RetinaSegmentationDataset, BasicSegmentationDataset
from torch.utils.data import DataLoader
from kornia.losses import focal_loss

import segmentation_models_pytorch.segmentation_models_pytorch as smp
import pandas as pd
import matplotlib.pyplot as plt

def train_net(net,
              device,
              training_set,
              validation_set,
              dir_checkpoint,
              epochs=150,
              batch_size=2,
              lr=0.001,
              save_cp=True,
              img_scale=1,
              n_classes=3,
              n_channels=3, 
              augmentation_ratio = 0):

    train = training_set 
    val = validation_set
    n_train = len(train)
    n_val = len(val)
    
    # Reduced num_workers to 2 for Windows compatibility
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Effective batch size with augmentation
    batch_size_effective = (1 + augmentation_ratio) * batch_size

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size_effective}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size} (Effective: {batch_size_effective})
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Augmentation ratio: {augmentation_ratio}
        Classes:         {n_classes}
        Encoder:         Unfrozen (Training all layers)
        Mixed Precision: Enabled
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Use ReduceLROnPlateau to lower LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    best_val_loss = float('inf')

    # Initialize GradScaler for AMP
    scaler = GradScaler()

    # Initialize history dictionary for Excel and Plotting
    history = {
        'epoch': [],
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        # Training metrics
        'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_iou': [], 'train_dice': [],
        # Validation metrics
        'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_iou': [], 'val_dice': []
    }

    if save_cp:
        try:
            os.makedirs(dir_checkpoint, exist_ok=True)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                imgs = torch.cat(imgs, dim = 0)
                true_masks = torch.cat(true_masks, dim = 0)

                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with autocast():
                    masks_pred = net(imgs)                         
                    
                    if n_classes == 1:
                        bce_loss = nn.BCEWithLogitsLoss()
                        loss = bce_loss(masks_pred, true_masks)
                        loss += dice_loss(masks_pred, true_masks).mean()
                    else:
                        loss = focal_loss(masks_pred, true_masks.squeeze(1), alpha=0.25, gamma=2, reduction='mean').unsqueeze(0)
                        loss += dice_loss(masks_pred, true_masks.squeeze(1), True, k=0.75)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual.
                nn.utils.clip_grad_value_(net.parameters(), 0.1)

                # optimizer.step() is replaced by scaler.step(optimizer)
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                pbar.update(imgs.shape[0]//(1 + augmentation_ratio))
                global_step += 1

        # --- End of Epoch Evaluation ---
        
        # 1. Validation Metrics
        val_metrics = eval_net(net, val_loader, device, n_classes)
        val_score = val_metrics['loss']
        
        # 2. Training Metrics 
        # (Evaluating on training set to get the graph points you requested)
        logging.info("Calculating training metrics (this may take a moment)...")
        # We temporarily switch to eval mode for metric calculation to be accurate
        train_metrics_full = eval_net(net, train_loader, device, n_classes)
        
        # Scheduler step based on validation loss
        scheduler.step(val_score)

        # 3. Log to History
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch + 1)
        history['learning_rate'].append(current_lr)
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / (n_train // (batch_size / (1 + augmentation_ratio)))
        history['train_loss'].append(avg_train_loss) 
        history['val_loss'].append(val_score)

        # Helper to safely extract metrics
        def extract_metrics(metric_dict, prefix, hist_dict):
            hist_dict[f'{prefix}_acc'].append(metric_dict.get('accuracy', 0))
            hist_dict[f'{prefix}_prec'].append(metric_dict.get('precision', 0))
            hist_dict[f'{prefix}_rec'].append(metric_dict.get('recall', 0))
            hist_dict[f'{prefix}_iou'].append(metric_dict.get('iou', 0))
            
            # Use dice if available, else calculate F1
            if 'dice' in metric_dict:
                hist_dict[f'{prefix}_dice'].append(metric_dict['dice'])
            else:
                p = metric_dict.get('precision', 0)
                r = metric_dict.get('recall', 0)
                dice = 2 * p * r / (p + r + 1e-8)
                hist_dict[f'{prefix}_dice'].append(dice)

        extract_metrics(train_metrics_full, 'train', history)
        extract_metrics(val_metrics, 'val', history)

        logging.info(f"Epoch {epoch+1} Results:")
        logging.info(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_score:.4f}")
        logging.info(f"  Train IoU:  {train_metrics_full.get('iou',0):.4f} | Val IoU:  {val_metrics.get('iou',0):.4f}")

        # Tensorboard
        writer.add_scalar('learning_rate', current_lr, global_step)
        writer.add_scalar('Loss/test', val_score, global_step)
        if 'iou' in val_metrics:
            writer.add_scalar('IoU/test', val_metrics['iou'], global_step)

        # Save Best Checkpoint
        if save_cp and val_score < best_val_loss:
            best_val_loss = val_score
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best.pth'))
            logging.info(f'New best checkpoint saved! Loss: {best_val_loss:.4f}')

        # Save Last Checkpoint
        if save_cp:
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_last.pth'))
            logging.info(f'Last checkpoint saved.')

    writer.close()

    # --- Post-Training: Save Excel and Plots ---
    
    # 1. Save Excel
    df = pd.DataFrame(history)
    excel_path = os.path.join(dir_checkpoint, 'training_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    logging.info(f'Metrics saved to {excel_path}')

    # 2. Plotting
    try:
        plt.figure(figsize=(18, 5))
        
        # Plot 1: Training Loss
        plt.subplot(1, 3, 1)
        plt.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['epoch'], history['val_loss'], label='Val Loss', color='orange', linestyle='--')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot 2: Learning Rate
        plt.subplot(1, 3, 2)
        plt.plot(history['epoch'], history['learning_rate'], label='Learning Rate', color='green')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True)

        # Plot 3: Dice Coefficient
        plt.subplot(1, 3, 3)
        plt.plot(history['epoch'], history['val_dice'], label='Val Dice', color='red')
        plt.plot(history['epoch'], history['train_dice'], label='Train Dice', color='pink', linestyle='--')
        plt.title('Dice Coefficient over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coeff')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(dir_checkpoint, 'training_plot.jpg')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Training plot saved to {plot_path}')
    except Exception as e:
        logging.error(f"Failed to save plots: {e}")

def get_args():
    parser = argparse.ArgumentParser(description='EfficientUNet++ train script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='Specifies the dataset to be used', dest='dataset', required=True)
    parser.add_argument('-ti', '--training-images-dir', type=str, default=None, help='Training images directory', dest='train_img_dir')
    parser.add_argument('-tm', '--training-masks-dir', type=str, default=None, help='Training masks directory', dest='train_mask_dir')
    parser.add_argument('-vi', '--validation-images-dir', type=str, default=None, help='Validation images directory', dest='val_img_dir')
    parser.add_argument('-vm', '--validation-masks-dir', type=str, default=None, help='Validation masks directory', dest='val_mask_dir')
    parser.add_argument('-enc', '--encoder', metavar='ENC', type=str, default='timm-efficientnet-b0', help='Encoder to be used', dest='encoder')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150, help='Number of epochs', dest='epochs')
    # CHANGED: Default batch size set to 2
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', type=str, default=False, help='Load model from a .pth file', dest='load')
    parser.add_argument('-s', '--scale', metavar='S', type=float, default=1, help='Downscaling factor of the images', dest='scale')
    parser.add_argument('-a', '--augmentation-ratio', metavar='AR', type=int, default=0, help='Number of augmentation to be generated for each image in the dataset', dest='augmentation_ratio')
    parser.add_argument('-c', '--dir_checkpoint', type=str, default='checkpoints/', help='Directory to save the checkpoints', dest='dir_checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.dataset == 'DRIVE':
        n_classes = 2
    elif args.dataset == 'Coronary':
        n_classes = 3
    elif args.dataset == 'LeafDisease':
        n_classes = 1 

    # Instantiate EfficientUNet++
    net = smp.EfficientUnetPlusPlus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=n_classes)

    # --- UNFREEZING STEPS ---
    # We simply do NOT call net.encoder.eval() or set requires_grad=False.
    # We ensure the network is in training mode.
    net.train()
    # ------------------------

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    # Instantiate datasets
    if args.dataset == 'DRIVE':
        training_set = RetinaSegmentationDataset(args.train_img_dir if args.train_img_dir is not None else 'DRIVE/training/images/', 
            args.train_mask_dir if args.train_mask_dir is not None else 'DRIVE/training/1st_manual/', args.scale, 
            augmentation_ratio = args.augmentation_ratio, crop_size=512)
        validation_set = RetinaSegmentationDataset(args.val_img_dir if args.val_img_dir is not None else 'DRIVE/validation/images/', 
            args.val_mask_dir if args.val_mask_dir is not None else 'DRIVE/validation/1st_manual/', args.scale)
    elif args.dataset == 'Coronary':
        training_set = CoronaryArterySegmentationDataset(args.train_img_dir if args.train_img_dir is not None else 'Coronary/train/imgs/', 
            args.train_mask_dir if args.train_mask_dir is not None else 'Coronary/train/masks/', args.scale, 
            augmentation_ratio = args.augmentation_ratio, crop_size=512)
        validation_set = CoronaryArterySegmentationDataset(args.val_img_dir if args.val_img_dir is not None else 'Coronary/val/imgs/', 
            args.val_mask_dir if args.val_mask_dir is not None else 'Coronary/val/masks/', args.scale, mask_suffix='a')
    elif args.dataset == 'LeafDisease':
        training_set = BasicSegmentationDataset(
            imgs_dir=args.train_img_dir, 
            masks_dir=args.train_mask_dir, 
            scale=args.scale,
            mask_suffix='' 
        )
        validation_set = BasicSegmentationDataset(
            imgs_dir=args.val_img_dir, 
            masks_dir=args.val_mask_dir, 
            scale=args.scale,
            mask_suffix=''
        )
    else:
        print("Invalid dataset")
        exit()

    try:
        train_net(net=net, 
                  device=device,
                  training_set=training_set,
                  validation_set=validation_set,
                  dir_checkpoint=args.dir_checkpoint,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  img_scale=args.scale,
                  n_classes=n_classes,
                  n_channels=3,
                  augmentation_ratio = args.augmentation_ratio)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)