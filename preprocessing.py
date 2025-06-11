import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch


input_dir = 'dataset/...'         
output_dir = 'dataset/...'       
image_size = 160                           
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mtcnn = MTCNN(image_size=image_size, margin=20, post_process=True, device=device)

for person_name in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    save_dir = os.path.join(output_dir, person_name)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        save_path = os.path.join(save_dir, img_name)

        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)

            if face is not None:
                face = face.permute(1, 2, 0).add(1).div(2).mul(255).clamp(0, 255).byte().cpu().numpy()
                Image.fromarray(face).save(save_path)
                print(f"✅ Saved: {save_path}")
            else:
                print(f"No face detected: {img_path}")

        except Exception as e:
            print(f"Error: {img_path} => {e}")
