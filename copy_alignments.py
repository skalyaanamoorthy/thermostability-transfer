import os
import shutil

def copy_files(src_dir, dst_dir):
    """Copies files from src_dir to dst_dir if first 4 letters of filename matches."""
    
    # List all files in the source directory
    src_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    print(src_files)

    # List all files in the destination directory
    dst_files = [f for f in os.listdir(dst_dir) if os.path.isdir(os.path.join(dst_dir, f))]
    print(dst_files)

    # For each file in the source directory...
    for src_file in src_files:
        # Check if there is a matching file in the destination directory
        for dst_file in dst_files:
            if src_file[:4] == dst_file[:4]:
                # If a matching file is found, copy the file from the source directory to the destination directory
                shutil.copy(os.path.join(src_dir, src_file), os.path.join(dst_dir, dst_file))
                print(f"Copied {src_file} to {dst_dir}")

# Call the function with your specific source and destination directories
copy_files("/mnt/d/original_subsampled_alignments", "./predictions")
