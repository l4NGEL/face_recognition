import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
from torchvision import transforms
from facenet_pytorch import MTCNN

class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, embedding_size)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [nn.Conv2d(in_c, out_c, 3, stride, 1),
                  nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers += [nn.Conv2d(out_c, out_c, 3, 1, 1),
                       nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
model = FaceNet(embedding_size=128).to(device)

if os.path.exists("best_facenet_model (5).pth"):
    model.load_state_dict(torch.load("best_facenet_model (5).pth", map_location=device))
    model.eval()
else:
    messagebox.showerror("Model Eksik", "Model dosyası (best_facenet_model.pth) bulunamadı!")


def add_new_person():
    name = simpledialog.askstring("Yeni Kişi", "Kişinin adını ve soyadını girin:")
    if not name: return
    os.makedirs("known_person", exist_ok=True)
    person_dir = os.path.join("known_person", name)
    os.makedirs(person_dir, exist_ok=True)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    count, frame_interval, frame_counter = 0, 5, 0
    messagebox.showinfo("Yüz Kaydı", "Kameraya bakın. Sistem 5 yüz görüntünüzü kaydedecek.")
    while count < 5:
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        face = mtcnn(img_pil)
        if face is not None:
            face = (face.clamp(-1, 1) + 1) / 2
            face_img = transforms.ToPILImage()(face).convert("RGB")

            frame_counter += 1
            if frame_counter % frame_interval == 0:
                filename = os.path.join(person_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                face_img.save(filename)
                count += 1
                print(f"📸 {count}/5 yüz görüntüsü kaydedildi.")
        cv2.imshow("Yüz Kaydi ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()
    if count > 0:
        messagebox.showinfo("Tamamlandı", f"{name} için {count} yüz görüntüsü kaydedildi!")
    else:
        messagebox.showwarning("Başarısız", "Yüz algılanamadı.")


def load_known_faces(folder="known_person"):
    known_embeddings = {}
    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        if not os.path.isdir(person_path): continue
        embeddings = []
        for img_name in os.listdir(person_path):
            try:
                img_path = os.path.join(person_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()[0]
                embeddings.append(emb)
            except: continue
        if embeddings:
            known_embeddings[person] = np.mean(embeddings, axis=0)
    return known_embeddings


def start_face_recognition():
    threading.Thread(target=recognize_faces).start()

def recognize_faces():
    known_faces = load_known_faces()
    if not known_faces:
        messagebox.showwarning("Veri Yok", "Tanınacak kayıtlı yüz bulunamadı!")
        return

    cap = cv2.VideoCapture(0)
    print("📷 Kamera başlatıldı. Yüz tanıma aktif...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        boxes, probs = mtcnn.detect(img_pil)
        faces = mtcnn(img_pil)
        if boxes is not None and faces is not None:
            for box, face in zip(boxes, faces):
                if face is None: continue
                face = (face.clamp(-1, 1) + 1) / 2
                face_img = transforms.ToPILImage()(face)
                face_tensor = transform(face_img.convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(face_tensor).cpu().numpy()[0]
                min_dist, best_match = float('inf'), "Bilinmiyor"
                for name, ref_emb in known_faces.items():
                    dist = np.linalg.norm(emb - ref_emb)
                    if dist < min_dist:
                        min_dist, best_match = dist, name
                if min_dist < 0.7:
                    label = f"✅ {best_match} tanindi. Kapi acildi."
                    color = (0, 255, 0)  
                else:
                    label = "❌ Taninmiyor. Kapi kapali."
                    color = (0, 0, 255)  


                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Yüz Tanıma (q ile çık)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()


root = tk.Tk()
root.title("Yüz Tanıma Sistemi")
root.geometry("400x300")
root.configure(bg="#1e1e1e")

tk.Label(root, text="Yüz Tanıma Sistemi", font=("Arial", 18), fg="white", bg="#1e1e1e").pack(pady=30)
tk.Button(root, text="📷 Kayıtlı Verilerle Yüz Tanıma", font=("Arial", 14), command=start_face_recognition).pack(pady=20)
tk.Button(root, text="➕ Yeni Kişi Ekle", font=("Arial", 14), command=add_new_person).pack(pady=10)
tk.Button(root, text="Çıkış", font=("Arial", 12), command=root.destroy).pack(pady=10)

root.mainloop()