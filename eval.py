# # import torch
# # import torch.nn.functional as F
# # from tqdm import tqdm

# # from metrics import dice_loss

# # def eval_net(net, loader, device, n_classes=3):
# #     net.eval()
# #     mask_type = torch.float32 if n_classes == 1 else torch.long
# #     n_val = len(loader)
# #     tot = 0

# #     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
# #         for batch in loader:
# #             imgs, true_masks = batch['image'][0], batch['mask'][0]
# #             imgs = imgs.to(device=device, dtype=torch.float32)
# #             true_masks = true_masks.to(device=device, dtype=mask_type)

# #             with torch.no_grad():
# #                 mask_pred = net(imgs)

# #             if n_classes > 1:
# #                 true_masks = true_masks.squeeze(1)
# #                 tot += dice_loss(mask_pred, true_masks, use_weights=True).item()
# #             else:
# #                 tot += dice_loss(mask_pred, true_masks, use_weights=False).item()
# #             pbar.update()

# #     net.train()
# #     return tot / n_val

# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from metrics import dice_loss

# def eval_net(net, loader, device, n_classes=3):
#     net.eval()
#     mask_type = torch.float32 if n_classes == 1 else torch.long
#     n_val = len(loader)
#     tot_loss = 0
    
#     # Initialize metric counters
#     tot_inter = 0
#     tot_union = 0
#     tot_tp = 0
#     tot_fp = 0
#     tot_fn = 0
#     tot_tn = 0

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#         for batch in loader:
#             imgs, true_masks = batch['image'][0], batch['mask'][0]
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             true_masks = true_masks.to(device=device, dtype=mask_type)

#             with torch.no_grad():
#                 mask_pred = net(imgs)

#             # Calculate Dice Loss
#             if n_classes > 1:
#                 true_masks = true_masks.squeeze(1)
#                 tot_loss += dice_loss(mask_pred, true_masks, use_weights=True).item()
#             else:
#                 tot_loss += dice_loss(mask_pred, true_masks, use_weights=False).item()
                
#                 # --- ADDED: Binary Metrics Calculation ---
#                 # Apply sigmoid to get probabilities and threshold at 0.5
#                 pred = torch.sigmoid(mask_pred) > 0.5
#                 target = true_masks > 0.5
                
#                 # Flatten
#                 pred = pred.view(-1)
#                 target = target.view(-1)

#                 # True Positives, False Positives, False Negatives, True Negatives
#                 tp = (pred & target).sum().item()
#                 fp = (pred & ~target).sum().item()
#                 fn = (~pred & target).sum().item()
#                 tn = (~pred & ~target).sum().item()

#                 tot_tp += tp
#                 tot_fp += fp
#                 tot_fn += fn
#                 tot_tn += tn
                
#                 # Intersection and Union for IoU
#                 intersection = (pred & target).sum().item()
#                 union = (pred | target).sum().item()
#                 tot_inter += intersection
#                 tot_union += union
                
#             pbar.update()

#     net.train()
    
#     # Calculate final metrics for the epoch
#     metrics = {'loss': tot_loss / n_val}
    
#     if n_classes == 1:
#         epsilon = 1e-7
#         precision = tot_tp / (tot_tp + tot_fp + epsilon)
#         recall = tot_tp / (tot_tp + tot_fn + epsilon)
#         accuracy = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn + epsilon)
#         iou = tot_inter / (tot_union + epsilon)
        
#         metrics['accuracy'] = accuracy
#         metrics['precision'] = precision
#         metrics['recall'] = recall
#         metrics['iou'] = iou

#     return metrics

# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from metrics import dice_loss

# def eval_net(net, loader, device, n_classes=3):
#     net.eval()
#     mask_type = torch.float32 if n_classes == 1 else torch.long
#     n_val = len(loader)
#     tot_loss = 0
    
#     # Initialize metric counters
#     tot_inter = 0
#     tot_union = 0
#     tot_tp = 0
#     tot_fp = 0
#     tot_fn = 0
#     tot_tn = 0

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#         for batch in loader:
#             imgs, true_masks = batch['image'][0], batch['mask'][0]
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             true_masks = true_masks.to(device=device, dtype=mask_type)

#             with torch.no_grad():
#                 mask_pred = net(imgs)

#             # Calculate Loss
#             if n_classes > 1:
#                 true_masks = true_masks.squeeze(1)
#                 tot_loss += dice_loss(mask_pred, true_masks, use_weights=True).item()
#             else:
#                 tot_loss += dice_loss(mask_pred, true_masks, use_weights=False).item()
                
#                 # Apply Sigmoid to logits to get probabilities for Dice calculation
#                 probs = torch.sigmoid(mask_pred)
#                 pred = probs > 0.5
#                 target = true_masks > 0.5
                
#                 # Intersection and Union for correct Dice/IoU calculation
#                 pred_flat = pred.view(-1)
#                 target_flat = target.view(-1)

#                 tp = (pred_flat & target_flat).sum().item()
#                 fp = (pred_flat & ~target_flat).sum().item()
#                 fn = (~pred_flat & target_flat).sum().item()
#                 tn = (~pred_flat & ~target_flat).sum().item()

#                 tot_tp += tp
#                 tot_fp += fp
#                 tot_fn += fn
#                 tot_tn += tn
                
#                 tot_inter += tp
#                 tot_union += (pred_flat | target_flat).sum().item()
                
#             pbar.update()

#     net.train()
    
#     metrics = {'loss': tot_loss / n_val}
    
#     if n_classes == 1:
#         epsilon = 1e-7
#         metrics['dice'] = (2. * tot_inter + epsilon) / (tot_tp + tot_fn + tot_tp + tot_fp + epsilon)
#         metrics['iou'] = tot_inter / (tot_union + epsilon)
#         metrics['accuracy'] = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn + epsilon)
#         metrics['precision'] = tot_tp / (tot_tp + tot_fp + epsilon)
#         metrics['recall'] = tot_tp / (tot_tp + tot_fn + epsilon)

#     return metrics

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from metrics import dice_loss
from kornia.losses import focal_loss

def predict_sliding_window_eval(model, image_tensor, patch_size=256, stride=128, num_classes=1, device='cuda'):
    """Helper for eval loop to do sliding window"""
    B, C, H, W = image_tensor.shape
    
    pred_canvas = torch.zeros((B, num_classes, H, W), device=device)
    count_canvas = torch.zeros((B, num_classes, H, W), device=device)

    y_steps = list(range(0, max(1, H - patch_size), stride)) + [max(0, H - patch_size)]
    x_steps = list(range(0, max(1, W - patch_size), stride)) + [max(0, W - patch_size)]
    
    y_steps = sorted(list(set(y_steps)))
    x_steps = sorted(list(set(x_steps)))

    for y in y_steps:
        for x in x_steps:
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            logits = model(patch)
            
            if num_classes == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
                
            pred_canvas[:, :, y:y+patch_size, x:x+patch_size] += probs
            count_canvas[:, :, y:y+patch_size, x:x+patch_size] += 1

    final_probs = pred_canvas / count_canvas
    if num_classes == 1:
        final_mask = (final_probs > 0.5).float()
    else:
        final_mask = torch.argmax(final_probs, dim=1).unsqueeze(1).float()
        
    return final_mask, final_probs

def eval_net(net, loader, device, n_classes=3):
    net.eval()
    mask_type = torch.float32 if n_classes == 1 else torch.long
    n_val = len(loader)
    tot_loss = 0
    
    tot_inter = 0
    tot_union = 0
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    tot_tn = 0
    
    bce_loss_fn = nn.BCEWithLogitsLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                pred_mask, pred_probs = predict_sliding_window_eval(net, imgs, patch_size=256, stride=128, num_classes=n_classes, device=device)

            if n_classes > 1:
                true_masks = true_masks.squeeze(1)
                # Approximate logits to match training loss behavior and avoid double-softmax in dice_loss
                logits_approx = torch.log(pred_probs.clamp(1e-6, 1-1e-6))
                loss = focal_loss(logits_approx, true_masks, alpha=0.25, gamma=2, reduction='mean').unsqueeze(0)
                loss += dice_loss(logits_approx, true_masks, True, k=0.75)
                tot_loss += loss.item()
            else:
                logits_approx = torch.logit(pred_probs.clamp(1e-6, 1-1e-6))
                tot_loss += bce_loss_fn(logits_approx, true_masks).item()
                tot_loss += dice_loss(logits_approx, true_masks, use_weights=False).item()
                
                # --- FIX: Convert floats to boolean before bitwise operations ---
                pred_flat = pred_mask.view(-1).bool()
                target_flat = (true_masks.view(-1) > 0.5).bool()

                tp = (pred_flat & target_flat).sum().item()
                fp = (pred_flat & ~target_flat).sum().item()
                fn = (~pred_flat & target_flat).sum().item()
                tn = (~pred_flat & ~target_flat).sum().item()

                tot_tp += tp
                tot_fp += fp
                tot_fn += fn
                tot_tn += tn
                
                tot_inter += tp
                tot_union += (pred_flat | target_flat).sum().item()
                
            pbar.update()

    net.train()
    
    metrics = {'loss': tot_loss / n_val}
    
    if n_classes == 1:
        epsilon = 1e-7
        metrics['dice'] = (2. * tot_inter + epsilon) / (tot_tp + tot_fn + tot_tp + tot_fp + epsilon)
        metrics['iou'] = tot_inter / (tot_union + epsilon)
        metrics['accuracy'] = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn + epsilon)
        metrics['precision'] = tot_tp / (tot_tp + tot_fp + epsilon)
        metrics['recall'] = tot_tp / (tot_tp + tot_fn + epsilon)

    return metrics