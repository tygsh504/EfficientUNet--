import os
from PIL import Image

def create_paired_patches(image_dir, mask_dir, out_image_dir, out_mask_dir, patch_size=256, overlap=0.5):
    """
    Cuts images and masks into identical patches, saving them as .png with the EXACT same filename.
    """
    # Create the output directories if they don't exist
    for directory in [out_image_dir, out_mask_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    stride = int(patch_size * (1 - overlap))
    if stride == 0:
        raise ValueError("Overlap cannot be 1.0 (100%).")

    # Map base names to full mask filenames
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))}

    for img_filename in os.listdir(image_dir):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            base_name, _ = os.path.splitext(img_filename)
            
            if base_name not in mask_files:
                print(f"⚠️ Skipping {img_filename}: No corresponding mask found.")
                continue
                
            mask_filename = mask_files[base_name]
            
            img_path = os.path.join(image_dir, img_filename)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            try:
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                
                if img.size != mask.size:
                    print(f"⚠️ Skipping {img_filename}: Image size {img.size} != Mask size {mask.size}")
                    continue

                width, height = img.size

                if width < patch_size or height < patch_size:
                    print(f"Skipping {img_filename}: Too small.")
                    continue

                x_starts = list(range(0, width - patch_size + 1, stride))
                y_starts = list(range(0, height - patch_size + 1, stride))

                if x_starts[-1] + patch_size < width:
                    x_starts.append(width - patch_size)
                if y_starts[-1] + patch_size < height:
                    y_starts.append(height - patch_size)

                patch_count = 0
                for y in y_starts:
                    for x in x_starts:
                        box = (x, y, x + patch_size, y + patch_size)
                        
                        img_patch = img.crop(box)
                        mask_patch = mask.crop(box)

                        # Force the output format to be .png for lossless saving
                        patch_filename = f"{base_name}_{x}_{y}.png"
                        
                        img_patch.save(os.path.join(out_image_dir, patch_filename), format="PNG")
                        mask_patch.save(os.path.join(out_mask_dir, patch_filename), format="PNG")
                        
                        patch_count += 1

                print(f"Processed: {img_filename} -> Created {patch_count} paired patches")

            except Exception as e:
                print(f"❌ Error processing {img_filename}: {e}")

# ==========================================
# Configuration Area
# ==========================================
if __name__ == "__main__":
    INPUT_IMAGES = r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\val\imgs"
    INPUT_MASKS = r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\val\masks"
    
    OUTPUT_IMAGES = r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\val\imgsc"
    OUTPUT_MASKS = r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\val\masksc"
    
    create_paired_patches(
        image_dir=INPUT_IMAGES,
        mask_dir=INPUT_MASKS,
        out_image_dir=OUTPUT_IMAGES,
        out_mask_dir=OUTPUT_MASKS,
        patch_size=256, 
        overlap=0.5 
    )