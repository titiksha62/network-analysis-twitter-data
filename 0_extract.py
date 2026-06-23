import tarfile
import os

source_file = "PHEME_veracity.tar.bz2" # Keep the name as is
target_dir = "./pheme_data"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f"Opening {source_file} as a GZIP archive...")

try:
    # 'r:*' will automatically detect that it's GZIP despite the .bz2 extension
    with tarfile.open(source_file, "r:*") as tar:
        tar.extractall(path=target_dir)
        print(f"Successfully extracted to {target_dir}")
        
        # List the first few items to confirm
        extracted_items = os.listdir(target_dir)
        print(f"Folders found: {extracted_items[:5]}")

except Exception as e:
    print(f"Extraction failed: {e}")