import os
import shutil

def copy_large_text_files(src_dir, dest_dir, min_size=500):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Walk through the source directory and its subdirectories
    for root, dirs, files in os.walk(src_dir):
        # Check each file in the directory
        for file in files:
            # Only process text files
            if file.endswith('.txt'):
                # Get the full path to the file
                file_path = os.path.join(root, file)
                
                # Check if the file is large enough
                if os.path.getsize(file_path) >= min_size:
                    # Copy the file to the destination directory
                    shutil.copy2(file_path, dest_dir)

# Example usage
copy_large_text_files(r'C:\Users\kapel\Desktop\news_3', r'C:\Users\kapel\Desktop\dataset')