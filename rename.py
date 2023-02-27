import os

def rename_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                old_filepath = os.path.join(dirpath, filename)
                new_filename = os.path.join(os.path.basename(root_dir), os.path.relpath(old_filepath, root_dir)).replace(os.path.sep, '_')
                new_filepath = os.path.join(dirpath, new_filename)
                os.rename(old_filepath, new_filepath)

# Example usage
rename_files(r'C:\Users\kapel\Desktop\news_3')
