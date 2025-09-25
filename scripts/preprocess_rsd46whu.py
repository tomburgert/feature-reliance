import os
import io

from PIL import Image
from tqdm import tqdm
import pandas as pd


# Class mapping
CLASS_NAMES = [
    "Airplane", "Airport", "Artificial dense forest land", "Artificial sparse forest land", "Bare land",
    "Basketball court", "Blue structured factory building", "Building", "Construction site", "Cross river bridge",
    "Crossroads", "Dense tall building", "Dock", "Fish pond", "Footbridge", "Graff", "Grassland",
    "Low scattered building", "Irregular farmland", "Medium density scattered building",
    "Medium density structured building", "Natural dense forest land", "Natural sparse forest land",
    "Oil tank", "Overpass", "Parting lot", "Plastic greenhouse", "Playground", "Railway",
    "Red structured factory building", "Refinery", "Regular farmland", "Scattered blue roof factory building",
    "Scattered red roof factory building", "Sewage plant-type-one", "Sewage plant-type-two", "Ship",
    "Solar power station", "Sparse residential area", "Square", "Steelworks", "Storage land", "Tennis court",
    "Thermal power plant", "Vegetable plot", "Water"
]


def save_images_from_parquet(parquet_path, output_dir):
    # Load the parquet file
    df = pd.read_parquet(parquet_path)

    # Create class folders
    for class_name in CLASS_NAMES:
        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # Iterate through each row
    for idx in tqdm(range(len(df)), desc="Saving images"):
        img_bytes = df.iloc[idx]['image']['bytes']
        label_id = df.iloc[idx]['label']
        class_name = CLASS_NAMES[label_id]

        # Decode the image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Create filename
        filename = f"{class_name}_{idx:06d}.png"
        filepath = os.path.join(output_dir, class_name, filename)

        # Save image
        image.save(filepath, format="PNG")


base_path = '/data/tomburgert/data/datasets/RSD46-WHU'

# process partquet file 1
path = os.path.join(base_path, '/parquet_files/train-00000-of-00004-27e4fea5a3d6f122.parquet')
save_images_from_parquet(path, os.path.join(base_path, 'images'))

# process partquet file 2
path = os.path.join(base_path, '/parquet_files/train-00001-of-00004-9d6b98a42da0cd9c.parquet')
save_images_from_parquet(path, os.path.join(base_path, 'images'))

# process partquet file 3
path = os.path.join(base_path, '/parquet_files/train-00002-of-00004-4450644e4b0b5cf2.parquet')
save_images_from_parquet(path, os.path.join(base_path, 'images'))

# process partquet file 4
path = os.path.join(base_path, '/parquet_files/train-00003-of-00004-1f0f07cbfc2667e5.parquet')
save_images_from_parquet(path, os.path.join(base_path, 'images'))
