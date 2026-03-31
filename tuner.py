import argparse
import logging
import os
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

# Import your existing modules
from metrics import dice_loss
from eval import eval_net
from utils.dataset import PatchSegmentationDataset
import segmentation_models_pytorch.segmentation_models_pytorch as smp

# Ensure Kornia focal loss is available as in your original script
from kornia.losses import focal_loss

def objective(trial):
    """
    Optuna objective function. This function trains the model with a specific 
    set of hyperparameters and returns the validation metric (e.g., validation loss).
    """
    # -------------------------------------------------------------------------
    # 1. Define Search Space (Hyperparameters to tune)
    # -------------------------------------------------------------------------
    # Log-uniform learning rate between 1e-5 and 1e-2
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    # Categorical batch size (constrained by GPU VRAM, 8 max is safe for an 8GB RTX 4060)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    
    # Try different lightweight encoders suitable for edge deployment
    encoder_name = trial.suggest_categorical("encoder", [
        "timm-efficientnet-b0", 
        "timm-efficientnet-b1", 
        "timm-mobilenetv3_large_100"
    ])
    
    # Optionally tune the optimizer type
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    # -------------------------------------------------------------------------
    # 2. Setup Device, Model, and Optimizer
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 1 # Assuming LeafDisease configuration

    net = smp.EfficientUnetPlusPlus(
        encoder_name=encoder_name, 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=n_classes
    ).to(device)

    # Initialize optimizer based on Optuna's suggestion
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    # -------------------------------------------------------------------------
    # 3. Setup Data Loaders (Using LeafDisease configuration)
    # -------------------------------------------------------------------------
    # We use overlapping patches for training, no overlap for validation
    train_set = PatchSegmentationDataset(
        imgs_dir=args.train_img_dir, 
        masks_dir=args.train_mask_dir, 
        patch_size=256, stride=128, mask_suffix='', is_train=True
    )
    val_set = PatchSegmentationDataset(
        imgs_dir=args.val_img_dir, 
        masks_dir=args.val_mask_dir, 
        patch_size=256, stride=256, mask_suffix='', is_train=False
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------------------------------------------------
    # 4. Streamlined Training Loop for Tuning
    # -------------------------------------------------------------------------
    epochs_for_tuning = args.tune_epochs

    for epoch in range(epochs_for_tuning):
        net.train()
        
        # Training pass
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_for_tuning}", leave=False)
        for batch in pbar:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                masks_pred = net(imgs)
                bce_loss = nn.BCEWithLogitsLoss()
                loss = bce_loss(masks_pred, true_masks)
                loss += dice_loss(masks_pred, true_masks).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()

        # Validation pass
        val_metrics = eval_net(net, val_loader, device, n_classes)
        val_loss = val_metrics['loss']

        # -------------------------------------------------------------------------
        # 5. Report to Optuna and check for Pruning
        # -------------------------------------------------------------------------
        # Report the validation metric at the end of the epoch
        trial.report(val_loss, epoch)

        # Handle pruning (stops trials early if they are performing poorly)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # We return val_loss because we want Optuna to minimize it. 
    # If you prefer to maximize IoU or Dice, return that and set direction="maximize" below.
    return val_loss

def get_args():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning')
    parser.add_argument('-ti', '--training-images-dir', type=str, required=True, dest='train_img_dir')
    parser.add_argument('-tm', '--training-masks-dir', type=str, required=True, dest='train_mask_dir')
    parser.add_argument('-vi', '--validation-images-dir', type=str, required=True, dest='val_img_dir')
    parser.add_argument('-vm', '--validation-masks-dir', type=str, required=True, dest='val_mask_dir')
    parser.add_argument('-te', '--tune-epochs', type=int, default=15, help='Epochs per trial', dest='tune_epochs')
    parser.add_argument('-n', '--n-trials', type=int, default=30, help='Total number of configurations to test')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    # Create an Optuna study. 
    # direction="minimize" because we are minimizing val_loss.
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    )
    
    logging.info(f"Starting hyperparameter optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)

    # Output the best results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest Trial:")
    trial = study.best_trial
    print(f"  Value (Val Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Optional: Save visualization of the tuning process
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("param_importances.png")
        print("Saved parameter importance plot to param_importances.png")
    except ImportError:
        pass