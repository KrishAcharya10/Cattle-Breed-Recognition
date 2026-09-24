import os
import shutil
import random

# Source and destination, relative to this project's root
source_folder = os.path.join(os.path.dirname(__file__), 'raw')      # one subfolder per breed/class
target_data_folder = os.path.join(os.path.dirname(__file__), 'data')  # will be created here

# Reproducible shuffling
random.seed(42)

# Create the base data directory and split subfolders
os.makedirs(target_data_folder, exist_ok=True)
splits = {'train': 0.70, 'val': 0.15, 'test': 0.15}
for split in splits:
    os.makedirs(os.path.join(target_data_folder, split), exist_ok=True)

# Iterate breeds/classes
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
for breed in os.listdir(source_folder):
    breed_folder = os.path.join(source_folder, breed)
    if not os.path.isdir(breed_folder):
        continue

    images = [f for f in os.listdir(breed_folder)
              if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(breed_folder, f))]
    total = len(images)
    if total == 0:
        continue

    random.shuffle(images)

    n_train = int(total * splits['train'])
    n_val   = int(total * splits['val'])
    n_test  = total - n_train - n_val  # remainder

    split_files = {
        'train': images[:n_train],
        'val'  : images[n_train:n_train + n_val],
        'test' : images[n_train + n_val:]
    }

    # Create breed subfolders in each split and copy files
    for split in splits:
        split_breed_folder = os.path.join(target_data_folder, split, breed)
        os.makedirs(split_breed_folder, exist_ok=True)
        for img_name in split_files[split]:
            src_path = os.path.join(breed_folder, img_name)
            dst_path = os.path.join(split_breed_folder, img_name)
            shutil.copy2(src_path, dst_path)  # preserves metadata

print("✅ Dataset split complete!")
print("Saved under:", os.path.abspath(target_data_folder))
