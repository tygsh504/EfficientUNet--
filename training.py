import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.amp  # Modern AMP namespace
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

# --- 1. COMPATIBILITY PATCH (From Notebook) ---
import collections.abc
import types
if 'torch._six' not in sys.modules:
    dummy_six = types.ModuleType('torch._six')
    dummy_six.container_abcs = collections.abc
    dummy_six.string_classes = (str, bytes)
    sys.modules['torch._six'] = dummy_six
    if not hasattr(torch, '_six'):
        torch._six = dummy_six

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
    
    # DataLoader optimized for Windows/RTX 4060
    loader_args = dict(
        batch_size=batch_size, 
        num_workers=2, 
        pin_memory=True
    )
    if loader_args['num_workers'] > 0:
        loader_args['persistent_workers'] = True
        loader_args['prefetch_factor'] = 2

    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)

    batch_size_effective = (1 + augmentation_ratio) * batch_size
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size_effective}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training (RTX 4060 Local):
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Mixed Precision: Enabled (Modern torch.amp)
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda') #

    history = {
        'epoch': [], 'learning_rate': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'train_pre': [], 'train_rec': [], 'train_iou': [], 'train_dice': [],
        'val_acc': [], 'val_pre': [], 'val_rec': [], 'val_iou': [], 'val_dice': []
    }

    if save_cp:
        os.makedirs(dir_checkpoint, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = torch.cat(batch['image'], dim = 0).to(device=device, dtype=torch.float32)
                true_masks = torch.cat(batch['mask'], dim = 0)
                # Force masks to float32 for proper Dice calculation in AMP
                mask_type = torch.float32 if n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'): #
                    masks_pred = net(imgs)                         
                    if n_classes == 1:
                        # Balanced loss to prevent 0-Dice
                        loss = nn.BCEWithLogitsLoss()(masks_pred, true_masks)
                        loss += dice_loss(masks_pred, true_masks).mean()
                    else:
                        loss = focal_loss(masks_pred, true_masks.squeeze(1), alpha=0.25, gamma=2, reduction='mean').unsqueeze(0)
                        loss += dice_loss(masks_pred, true_masks.squeeze(1), True, k=0.75)

                if loss.ndim > 0: loss = loss.mean()
                epoch_loss += loss.item()
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()

                pbar.update(imgs.shape[0]//(1 + augmentation_ratio))
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                global_step += 1

        # --- EVALUATION & CONSOLE LOGGING ---
        val_metrics = eval_net(net, val_loader, device, n_classes)
        train_metrics_full = eval_net(net, train_loader, device, n_classes)
        val_score = val_metrics['loss']
        scheduler.step(val_score)

        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch + 1)
        history['learning_rate'].append(current_lr)
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_score)

        def extract_metrics(metric_dict, prefix, hist_dict):
            for m in ['accuracy', 'precision', 'recall', 'iou']:
                hist_dict[f'{prefix}_{m[:3]}'].append(metric_dict.get(m, 0))
            hist_dict[f'{prefix}_dice'].append(metric_dict.get('dice', 0))

        extract_metrics(train_metrics_full, 'train', history)
        extract_metrics(val_metrics, 'val', history)

        logging.info(f'''
Epoch {epoch + 1} Summary:
-----------------------
Train Loss: {history['train_loss'][-1]:.4f} | Train Dice: {history['train_dice'][-1]:.4f}
Val Loss:   {history['val_loss'][-1]:.4f} | Val Dice:   {history['val_dice'][-1]:.4f}
-----------------------
''')

        if save_cp:
            # 1. Save and prompt for the Last Epoch
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_last.pth'))
            logging.info(f'Checkpoint: Last epoch ({epoch + 1}) weights saved to CP_last.pth')
            
            # 2. Save and prompt for the Best Epoch
            if val_score < best_val_loss:
                best_val_loss = val_score
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best.pth'))
                logging.info(f'Checkpoint: New BEST model saved to CP_best.pth (Val Loss: {val_score:.4f})')

    # Final saving logic
    pd.DataFrame(history).to_excel(os.path.join(dir_checkpoint, 'training_metrics.xlsx'), index=False)
    plt.figure(figsize=(18, 5))
    plt.subplot(1,3,1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,3,2); plt.plot(history['learning_rate']); plt.title('LR'); plt.yscale('log')
    plt.subplot(1,3,3); plt.plot(history['val_dice'], label='Val'); plt.plot(history['train_dice'], label='Train'); plt.title('Dice'); plt.legend()
    plt.savefig(os.path.join(dir_checkpoint, 'training_plot.jpg'))
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='EfficientUNet++ train script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='Specifies the dataset to be used', dest='dataset', required=True)
    parser.add_argument('-ti', '--training-images-dir', type=str, default=None, help='Training images directory', dest='train_img_dir')
    parser.add_argument('-tm', '--training-masks-dir', type=str, default=None, help='Training masks directory', dest='train_mask_dir')
    parser.add_argument('-vi', '--validation-images-dir', type=str, default=None, help='Validation images directory', dest='val_img_dir')
    parser.add_argument('-vm', '--validation-masks-dir', type=str, default=None, help='Validation masks directory', dest='val_mask_dir')
    parser.add_argument('-enc', '--encoder', metavar='ENC', type=str, default='timm-efficientnet-b0', help='Encoder to be used', dest='encoder')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', type=str, default=False, help='Load model from a .pth file', dest='load')
    parser.add_argument('-s', '--scale', metavar='S', type=float, default=1, help='Downscaling factor of the images', dest='scale')
    parser.add_argument('-a', '--augmentation-ratio', metavar='AR', type=int, default=0, help='Augmentation ratio', dest='augmentation_ratio')
    parser.add_argument('-c', '--dir_checkpoint', type=str, default='checkpoints/', help='Directory for checkpoints', dest='dir_checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_classes = 1 if args.dataset == 'LeafDisease' else (2 if args.dataset == 'DRIVE' else 3)
    net = smp.EfficientUnetPlusPlus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=n_classes)
    
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
    
    net.to(device)

    if args.dataset == 'LeafDisease':
        training_set = BasicSegmentationDataset(args.train_img_dir, args.train_mask_dir, args.scale)
        validation_set = BasicSegmentationDataset(args.val_img_dir, args.val_mask_dir, args.scale)

    train_net(net, device, training_set, validation_set, args.dir_checkpoint, args.epochs, args.batchsize, args.lr, True, args.scale, n_classes)