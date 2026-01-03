# import torch
# import torch.nn.functional as F
# from tqdm import tqdm

# from metrics import dice_loss

# def eval_net(net, loader, device, n_classes=3):
#     net.eval()
#     mask_type = torch.float32 if n_classes == 1 else torch.long
#     n_val = len(loader)
#     tot = 0

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#         for batch in loader:
#             imgs, true_masks = batch['image'][0], batch['mask'][0]
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             true_masks = true_masks.to(device=device, dtype=mask_type)

#             with torch.no_grad():
#                 mask_pred = net(imgs)

#             if n_classes > 1:
#                 true_masks = true_masks.squeeze(1)
#                 tot += dice_loss(mask_pred, true_masks, use_weights=True).item()
#             else:
#                 tot += dice_loss(mask_pred, true_masks, use_weights=False).item()
#             pbar.update()

#     net.train()
#     return tot / n_val

import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import dice_loss

def eval_net(net, loader, device, n_classes=3):
    net.eval()
    mask_type = torch.float32 if n_classes == 1 else torch.long
    n_val = len(loader)
    tot_loss = 0
    
    # Initialize metric counters
    tot_inter = 0
    tot_union = 0
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    tot_tn = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'][0], batch['mask'][0]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            # Calculate Dice Loss
            if n_classes > 1:
                true_masks = true_masks.squeeze(1)
                tot_loss += dice_loss(mask_pred, true_masks, use_weights=True).item()
            else:
                tot_loss += dice_loss(mask_pred, true_masks, use_weights=False).item()
                
                # --- ADDED: Binary Metrics Calculation ---
                # Apply sigmoid to get probabilities and threshold at 0.5
                pred = torch.sigmoid(mask_pred) > 0.5
                target = true_masks > 0.5
                
                # Flatten
                pred = pred.view(-1)
                target = target.view(-1)

                # True Positives, False Positives, False Negatives, True Negatives
                tp = (pred & target).sum().item()
                fp = (pred & ~target).sum().item()
                fn = (~pred & target).sum().item()
                tn = (~pred & ~target).sum().item()

                tot_tp += tp
                tot_fp += fp
                tot_fn += fn
                tot_tn += tn
                
                # Intersection and Union for IoU
                intersection = (pred & target).sum().item()
                union = (pred | target).sum().item()
                tot_inter += intersection
                tot_union += union
                
            pbar.update()

    net.train()
    
    # Calculate final metrics for the epoch
    metrics = {'loss': tot_loss / n_val}
    
    if n_classes == 1:
        epsilon = 1e-7
        precision = tot_tp / (tot_tp + tot_fp + epsilon)
        recall = tot_tp / (tot_tp + tot_fn + epsilon)
        accuracy = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn + epsilon)
        iou = tot_inter / (tot_union + epsilon)
        
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['iou'] = iou

    return metrics