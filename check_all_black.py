import os
from PIL import Image

def find_black_images(folder_path):
    """
    Scans a folder and returns a list of filenames that are completely black.
    """
    black_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return black_images

    print(f"Scanning folder: {folder_path}...\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(folder_path, filename)
            
            try:
                with Image.open(filepath) as img:
                    grayscale_img = img.convert('L')
                    
                    if grayscale_img.getbbox() is None:
                        black_images.append(filename)
                        
            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")

    return black_images

def delete_matching_images(black_images_list, first_folder_path, second_folder_path):
    """
    Prompts the user for confirmation and deletes the specified images from both folders.
    """
    print("\n" + "="*40)
    print("⚠️  WARNING: DELETION CANNOT BE UNDONE ⚠️")
    print("="*40)
    
    # Prompt the user for action
    user_input = input("\nReply 'Yes' to delete these images from BOTH folders. (Any other key will cancel): ")
    
    if user_input.strip().lower() in ['yes', 'y']:
        print("\nStarting deletion process...")
        
        for filename in black_images_list:
            path_first = os.path.join(first_folder_path, filename)
            path_second = os.path.join(second_folder_path, filename)
            
            # 1. Delete from the first folder (the all-black images)
            try:
                if os.path.exists(path_first):
                    os.remove(path_first)
                    print(f"Deleted: {path_first}")
            except Exception as e:
                print(f"Error deleting {path_first}: {e}")

            # 2. Delete from the second folder (the original images)
            try:
                if os.path.exists(path_second):
                    os.remove(path_second)
                    print(f"Deleted: {path_second}")
                else:
                    print(f"Skipped (not found): {path_second}")
            except Exception as e:
                print(f"Error deleting {path_second}: {e}")
                
        print("\n✅ Deletion complete.")
    else:
        print("\n❌ Action cancelled. No images were deleted.")

if __name__ == "__main__":
    # --- SETUP YOUR FOLDER PATHS HERE ---
    first_folder = r"C:\Users\User\Desktop\Paddy_Dataset\Combined\Training_GT"   # Folder containing the images to check for black
    second_folder = r"C:\Users\User\Desktop\Paddy_Dataset\Combined\Training_Ori" # Folder containing original images to remove
    
    # Run the detection
    results = find_black_images(first_folder)
    
    if results:
        print(f"Total black images found: {len(results)}")
        print("List of black images detected:")
        for name in results:
            print(f"- {name}")
            
        # Trigger the deletion prompt
        delete_matching_images(results, first_folder, second_folder)
    else:
        print("-" * 30)
        print("No completely black images were found in the first folder. Nothing to delete.")