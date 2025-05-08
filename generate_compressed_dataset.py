import os
from PIL import Image

SOURCE_DIRS = ["actors", "youtube"]
DEST_DIR = "compressed_dataset"
IMAGE_SIZE = (224, 224)

os.makedirs(DEST_DIR, exist_ok=True)

for source_dir in SOURCE_DIRS:
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(".jpg"):
                    continue
                img_path = os.path.join(folder_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) // 2
                    top = (height - min_dim) // 2
                    right = left + min_dim
                    bottom = top + min_dim
                    img = img.crop((left, top, right, bottom))
                    img = img.resize(IMAGE_SIZE)
                    new_filename = f"{source_dir}_{folder_name}_{img_name}"
                    img.save(os.path.join(DEST_DIR, new_filename))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
