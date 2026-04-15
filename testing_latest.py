import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.amp # Modern AMP namespace

# --- COMPATIBILITY PATCH FOR MODERN PYTORCH (2.0+) ---
import collections.abc
import types
if 'torch._six' not in sys.modules:
    dummy_six = types.ModuleType('torch._six')
    dummy_six.container_abcs = collections.abc
    dummy_six.string_classes = (str, bytes)
    sys.modules['torch._six'] = dummy_six
    if not hasattr(torch, '_six'):
        torch._six = dummy_six
# -----------------------------------------------------

# --- IMPORT MODEL ---
try:
    import segmentation_models_pytorch.segmentation_models_pytorch as smp
except ImportError:
    import segmentation_models_pytorch as smp

# =====================================================================
# --- USER CONFIGURATION SECTION ---
# =====================================================================
MODEL_PATH = 'checkpoints\CP_best.pth'
BASE_DATA_PATH = r"C:\Users\User\Desktop\testing dataset"
MAIN_OUTPUT_DIR = r"C:\Users\User\Desktop\b0_new_dataset_14"

# The 7 disease folders
DISEASES = ["Bacterial Leaf Blight", "Bacterial Leaf Streak", "Blast", "Brown Spot", "DownyMildew", "Hispa", "Tungro"]

# Model Config
ENCODER_NAME = 'timm-efficientnet-b0'
NUM_CLASSES = 1         
INPUT_SHAPE = [640, 480] # [Height, Width]
# =====================================================================

def calculate_complexity(model, input_shape, device):
    """Calculates model complexity (params and FLOPs)."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, *input_shape).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return params, flops
    except ImportError:
        logging.warning("Could not import 'thop'. FLOPs and Params will not be calculated.")
        return 0, 0
    except Exception as e:
        logging.error(f"Error during complexity calculation: {e}")
        return 0, 0

def calculate_metrics(pred_mask, true_mask):
    pred = pred_mask.detach().cpu().numpy().flatten()
    true = true_mask.detach().cpu().numpy().flatten()
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

    return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall, "Accuracy": accuracy, "F1_Score": f1}

def save_visual_result(image_tensor, true_mask_tensor, pred_mask_tensor, filename, dice_score, output_dir):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
    img_np = (img_np * 255).astype(np.uint8)
    
    true_np = true_mask_tensor.squeeze().cpu().numpy()
    pred_np = pred_mask_tensor.squeeze().cpu().numpy()
    true_bin = (true_np > 0.5).astype(np.uint8)
    pred_bin = (pred_np > 0.5).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_np); ax[0].set_title(f"Original: {filename}"); ax[0].axis("off")
    ax[1].imshow(true_bin, cmap='gray'); ax[1].set_title("Ground Truth"); ax[1].axis("off")
    ax[2].imshow(pred_bin, cmap='gray'); ax[2].set_title(f"Pred (Dice: {dice_score:.2f})"); ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_eval.png", dpi=150)
    plt.close(fig)

class SimpleDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, input_shape):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.input_shape = input_shape
        self.ids = [f for f in os.listdir(imgs_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = os.path.join(self.imgs_dir, idx)
        mask_name = idx if os.path.exists(os.path.join(self.masks_dir, idx)) else os.path.splitext(idx)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # PIL resize takes (Width, Height)
        target_size = (self.input_shape[1], self.input_shape[0]) 
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

        img_tensor = transforms.ToTensor()(img)
        mask_np = np.array(mask)
        if len(mask_np.shape) == 3: mask_np = mask_np[:, :, 0]
        mask_np = (mask_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {'image': img_tensor, 'mask': mask_tensor, 'name': idx}

def run_test_on_disease(disease_name, net, device, params, flops):
    img_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_Ori")
    mask_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_GT")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.warning(f"Skipping {disease_name}: Path not found.")
        return None

    disease_output_dir = Path(MAIN_OUTPUT_DIR) / disease_name
    img_output_dir = disease_output_dir / "predictions"
    disease_output_dir.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = SimpleDataset(img_dir, mask_dir, INPUT_SHAPE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    
    net.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            for batch in tqdm(test_loader, desc=f"Testing {disease_name}"):
                images = batch['image'].to(device, dtype=torch.float32)
                true_masks = batch['mask'].to(device, dtype=torch.float32)
                img_names = batch['name']

                outputs = net(images)
                
                if NUM_CLASSES == 1:
                    probs = torch.sigmoid(outputs)
                    pred_masks = (probs > 0.5).float()
                else:
                    probs = torch.softmax(outputs, dim=1)
                    pred_masks = torch.argmax(probs, dim=1).unsqueeze(1).float()

                metrics = calculate_metrics(pred_masks[0], true_masks[0])
                metrics['Filename'] = img_names[0]
                results.append(metrics)
                save_visual_result(images[0], true_masks[0], pred_masks[0], img_names[0], metrics['Dice'], img_output_dir)

    if results:
        df = pd.DataFrame(results)
        metric_cols = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
        means = df[metric_cols].mean().to_dict()
        summary_df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in means.items()] + 
                                  [{'Metric': 'Params', 'Value': params}, {'Metric': 'FLOPs', 'Value': flops}])

        excel_path = disease_output_dir / f'{disease_name}_metrics.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            df[['Filename'] + metric_cols].to_excel(writer, sheet_name='Detailed', index=False)
            
        return means
    return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        net = smp.EfficientUnetPlusPlus(
            encoder_name=ENCODER_NAME, 
            encoder_weights=None, 
            in_channels=3, 
            classes=NUM_CLASSES
        )
        
        # Safely load weights and handle nn.DataParallel "module." prefix from train_latest.py
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict)
        
        net.to(device)
        net.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model architecture: {e}")
        exit(1)

    params, flops = calculate_complexity(net, INPUT_SHAPE, device)

    all_disease_results = []

    for disease in DISEASES:
        disease_means = run_test_on_disease(disease, net, device, params, flops)
        if disease_means:
            disease_means['Disease'] = disease
            all_disease_results.append(disease_means)

    if all_disease_results:
        overall_df = pd.DataFrame(all_disease_results)
        cols = ['Disease'] + [c for c in overall_df.columns if c != 'Disease']
        overall_df = overall_df[cols]
        
        mean_row = overall_df.mean(numeric_only=True).to_dict()
        mean_row['Disease'] = 'OVERALL_MEAN'
        overall_df = pd.concat([overall_df, pd.DataFrame([mean_row])], ignore_index=True)
        
        Path(MAIN_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        mean_output_path = Path(MAIN_OUTPUT_DIR) / 'calculated_mean.xlsx'
        overall_df.to_excel(mean_output_path, index=False)
        logging.info(f"Overall means saved to {mean_output_path}")

    print("\n--- All Testing Completed ---")