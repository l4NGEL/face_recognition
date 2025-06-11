import pandas as pd
import os
import shutil
import random
from tqdm import tqdm

csv_path = "list_attr_celeba.csv"
input_root = "processed"
output_root = "known_faces"


df = pd.read_csv(csv_path)
df.set_index("image_id", inplace=True)

split_config = {
    "train": {"female": 40, "male": 40},
    "val": {"female": 10, "male": 10},
    "test": {"female": 10, "male": 10},
}


for split in ["train", "val", "test"]:
    print(f"\n🔍 {split.upper()} klasörü işleniyor...")
    gender_groups = {"female": [], "male": []}

    base_folder = os.path.join(input_root, split)
    for person in os.listdir(base_folder):
        person_path = os.path.join(base_folder, person)
        if not os.path.isdir(person_path):
            continue

        images = [f for f in os.listdir(person_path) if f.endswith(".jpg")]


        if len(images) < 5:
            continue


        gender_votes = []
        for img in images:
            if img in df.index:
                gender_votes.append(df.loc[img]["Male"])
        if gender_votes:
            majority_gender = pd.Series(gender_votes).mode()[0]
            gender = "female" if majority_gender == -1 else "male"
            gender_groups[gender].append(person)


    for gender in ["female", "male"]:
        random.shuffle(gender_groups[gender])
        selected_people = gender_groups[gender][:split_config[split][gender]]

        output_split_dir = os.path.join(output_root, split)
        os.makedirs(output_split_dir, exist_ok=True)

        for person in tqdm(selected_people, desc=f"{split}-{gender}"):
            src_path = os.path.join(base_folder, person)
            dst_path = os.path.join(output_split_dir, person)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            os.makedirs(dst_path)

            person_images = [f for f in os.listdir(src_path) if f.endswith(".jpg")]
            selected_images = random.sample(person_images,20)

            for img_name in selected_images:
                shutil.copy(
                    os.path.join(src_path, img_name),
                    os.path.join(dst_path, img_name)
                )

print("\n10 fotoğrafla kişiler başarıyla kopyalandı.")
