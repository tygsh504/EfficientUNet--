import os
from PIL import Image

# ==========================================
# CHANGE THIS TO YOUR VALIDATION IMAGE FOLDER PATH
# ==========================================
val_img_dir = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Val_GT"

def print_rgba_filenames(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory not found -> {directory}")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith('.'): continue # Skip hidden files
        
        filepath = os.path.join(directory, filename)
        
        try:
            with Image.open(filepath) as img:
                # Check ONLY for 4-channel images
                if img.mode == 'RGBA':
                    print(filename)
                    
        except Exception:
            pass # Skip files that aren't images

# Run the function
print_rgba_filenames(val_img_dir)