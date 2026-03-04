import os
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- IMPORT MODEL ---
# Using the specific import path from your train.py
try:
    import segmentation_models_pytorch.segmentation_models_pytorch as smp
except ImportError:
    # Fallback if running from a different directory structure
    import segmentation_models_pytorch as smp

# --- USER CONFIGURATION SECTION ---
#
MODEL_PATH = 'checkpoints/CP_best.pth' # Path to your best checkpoint

# SPECIFY THE PATH TO YOUR TESTING DATASET HERE
TEST_IMG_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_Ori"
TEST_MASK_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_GT"

# Model Config
ENCODER_NAME = 'timm-efficientnet-b0' # Must match what you used in train.py
NUM_CLASSES = 1         # 1 for Binary (Background vs Disease), 3 for Multiclass
INPUT_SHAPE = [640, 480] # [Height, Width]
# ----------------------------------

def calculate_complexity(model, input_shape, device):
    """Calculates Params and FLOPs."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return params, flops
    except Exception as e:
        logging.warning(f"Could not calculate complexity: {e}")
        return 0, 0

def calculate_metrics(pred_mask, true_mask):
    """Calculates binary metrics (Foreground vs Background)."""
    # Move to CPU and numpy
    pred = pred_mask.detach().cpu().numpy().flatten()
    true = true_mask.detach().cpu().numpy().flatten()
    
    # Binarize
    # Assuming pred_mask is already 0 or 1, but safety cast
    pred_bin = (pred > 0.5).astype(np.uint8)
    true_bin = (true > 0.5).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    tn = np.sum((pred_bin == 0) & (true_bin == 0))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))

    epsilon = 1e-7
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1_Score": f1
    }

def save_visual_result(image_tensor, true_mask_tensor, pred_mask_tensor, filename, dice_score, output_dir):
    """Saves a comparison image."""
    # Convert Image Tensor to PIL (Reverse normalization if needed, here assuming simple scaling)
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    # Normalize to 0-255 for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
    img_np = (img_np * 255).astype(np.uint8)
    
    true_np = true_mask_tensor.squeeze().cpu().numpy()
    pred_np = pred_mask_tensor.squeeze().cpu().numpy()

    # Binarize for visualization
    true_bin = (true_np > 0.5).astype(np.uint8)
    pred_bin = (pred_np > 0.5).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    ax[0].imshow(img_np)
    ax[0].set_title(f"Original: {filename}", fontsize=12)
    ax[0].axis("off")

    # Ground Truth (Black & White)
    ax[1].imshow(true_bin, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax[1].set_title("Ground Truth", fontsize=12)
    ax[1].axis("off")

    # Prediction (Black & White)
    ax[2].imshow(pred_bin, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax[2].set_title(f"Pred (Dice: {dice_score:.2f})", fontsize=12)
    ax[2].axis("off")

    plt.tight_layout()
    save_path = output_dir / f"{filename}_eval.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

class SimpleDataset(Dataset):
    """
    A simple dataset class to handle image/mask loading and resizing
    consistent with EfficientUNet++ requirements.
    """
    def __init__(self, imgs_dir, masks_dir, input_shape):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.input_shape = input_shape # [H, W]
        self.ids = [f for f in os.listdir(imgs_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = os.path.join(self.imgs_dir, idx)
        
        # Try to find matching mask (png or jpg)
        mask_name = idx
        if not os.path.exists(os.path.join(self.masks_dir, mask_name)):
             mask_name = os.path.splitext(idx)[0] + '.png'
        
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) # Mask might be grayscale or indexed

        # Resize
        # PIL resize takes (Width, Height)
        target_size = (self.input_shape[1], self.input_shape[0]) 
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

        # Transform to Tensor
        img_tensor = transforms.ToTensor()(img)
        mask_np = np.array(mask)
        
        # Handle mask values: Normalize to 0-1 (Binary) or Long (Multiclass)
        # Assuming binary 0-255 or 0-1 mask
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0] # Take 1 channel
            
        mask_np = (mask_np > 0).astype(np.float32) # Convert to binary float 0.0 or 1.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) # Add channel dim [1, H, W]

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'name': idx
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 1. Validation
    if not os.path.exists(TEST_IMG_DIR) or not os.path.exists(TEST_MASK_DIR):
        logging.error("Dataset paths not found.")
        exit(1)
        
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        exit(1)

    # 2. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 3. Load Model
    #
    try:
        logging.info(f"Loading EfficientUnetPlusPlus with encoder {ENCODER_NAME}...")
        net = smp.EfficientUnetPlusPlus(
            encoder_name=ENCODER_NAME, 
            encoder_weights=None, # Weights loaded from checkpoint
            in_channels=3, 
            classes=NUM_CLASSES
        )
        
        # Load State Dict
        # Handle DataParallel wrapping ('module.' prefix) if present in checkpoint
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        net.load_state_dict(new_state_dict)
        net.to(device)
        net.eval()
        logging.info("Model loaded successfully.")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        exit(1)

    # 4. Setup Data
    test_dataset = SimpleDataset(TEST_IMG_DIR, TEST_MASK_DIR, INPUT_SHAPE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Output Directories
    output_dir = Path("testing_output")
    img_output_dir = output_dir / "predictions"
    output_dir.mkdir(exist_ok=True)
    img_output_dir.mkdir(exist_ok=True)

    # 6. Calculate Complexity
    params, flops = calculate_complexity(net, INPUT_SHAPE, device)
    logging.info(f"Params: {params:,.0f} | FLOPs: {flops:,.0f}")

    results = []

    logging.info(f"Starting inference on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device, dtype=torch.float32)
            true_masks = batch['mask'].to(device, dtype=torch.float32)
            img_names = batch['name']

            # Inference
            # EfficientUNet returns logits
            outputs = net(images)
            
            # Post-processing
            if NUM_CLASSES == 1:
                probs = torch.sigmoid(outputs)
                pred_masks = (probs > 0.5).float()
            else:
                probs = torch.softmax(outputs, dim=1)
                pred_masks = torch.argmax(probs, dim=1).unsqueeze(1).float()

            # Calculate Metrics for batch (batch size is 1)
            for i in range(len(images)):
                metrics = calculate_metrics(pred_masks[i], true_masks[i])
                metrics['Filename'] = img_names[i]
                results.append(metrics)
                
                # Visuals
                save_visual_result(
                    images[i], 
                    true_masks[i], 
                    pred_masks[i], 
                    img_names[i], 
                    metrics['Dice'], 
                    img_output_dir
                )

    # 7. Save Excel
    if results:
        # Detailed
        df = pd.DataFrame(results)
        column_order = ['Filename', 'Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
        df = df[column_order]

        # Summary
        metric_cols = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
        means = df[metric_cols].mean().to_dict()
        
        summary_data = [
            {'Metric': 'Dice', 'Value': means['Dice']},
            {'Metric': 'IoU', 'Value': means['IoU']},
            {'Metric': 'Precision', 'Value': means['Precision']},
            {'Metric': 'Recall', 'Value': means['Recall']},
            {'Metric': 'Accuracy', 'Value': means['Accuracy']},
            {'Metric': 'F1', 'Value': means['F1_Score']},
            {'Metric': 'Params', 'Value': params},
            {'Metric': 'FLOPs', 'Value': flops}
        ]
        summary_df = pd.DataFrame(summary_data)

        excel_path = output_dir / 'performance_metrics.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
        logging.info(f"Testing Complete.")
        logging.info(f"Report saved to: {excel_path}")
        print("\n--- Summary ---")
        print(summary_df)
    else:
        logging.warning("No results generated. Check input paths.")