import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
from torchvision import transforms
from facenet_pytorch import MTCNN
import json


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


def load_person_database(path="person_database.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def add_new_person():
    name = simpledialog.askstring("Yeni Kişi", "Kişinin adını ve soyadını girin:")
    if not name:
        return

    
    tc = simpledialog.askstring("TC Kimlik No", f"{name} için TC Kimlik No:")
    birthdate = simpledialog.askstring("Doğum Tarihi", f"{name} için Doğum Tarihi (YYYY-AA-GG):")
    address = simpledialog.askstring("Adres", f"{name} için Adres bilgisi:")
    phone = simpledialog.askstring("Telefon", f"{name} için Telefon numarası:")

    
    db_path = "person_database.json"
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            database = json.load(f)
    else:
        database = {}

    database[name] = {
        "TC": tc,
        "Doğum Tarihi": birthdate,
        "Adres": address,
        "Telefon": phone
    }

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=4, ensure_ascii=False)

    
    os.makedirs("known_person", exist_ok=True)
    person_dir = os.path.join("known_person", name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count, frame_interval, frame_counter = 0, 5, 0
    messagebox.showinfo("Yüz Kaydı", "Kameraya bakın. Sistem 5 yüz görüntünüzü kaydedecek.")

    while count < 20:
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
        cv2.imshow("Yüz Kaydı", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

    if count > 0:
        messagebox.showinfo("Tamamlandı", f"{name} için {count} yüz görüntüsü ve kimlik bilgileri kaydedildi!")
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

def clean_orphan_database_entries():
    db_path = "person_database.json"
    face_folder = "known_person"

    if not os.path.exists(db_path) or not os.path.exists(face_folder):
        return

    with open(db_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    folder_names = set(os.listdir(face_folder))
    to_delete = [name for name in database if name not in folder_names]

    if to_delete:
        for name in to_delete:
            print(f"{name} klasörü yok, JSON'dan silindi.")
            del database[name]

        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=4, ensure_ascii=False)


def recognize_faces():
    
    clean_orphan_database_entries()

    known_faces = load_known_faces()
    database = load_person_database()
    recognized_once = set()

    if not known_faces:
        messagebox.showwarning("Veri Yok", "Tanınacak kayıtlı yüz bulunamadı!")
        return

    cap = cv2.VideoCapture(0)
    print("Kamera başlatıldı. Yüz tanıma aktif...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        boxes, probs = mtcnn.detect(img_pil)
        faces = mtcnn.extract(img_pil, boxes, save_path=None) if boxes is not None else None  # 👈 Çoklu yüz kırpma

        if boxes is not None and faces is not None:
            for box, face_img in zip(boxes, faces):
                if face_img is None:
                    continue

                face_tensor = transform(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model(face_tensor).cpu().numpy()[0]

                min_dist, best_match = float('inf'), None
                for name, ref_emb in known_faces.items():
                    dist = np.linalg.norm(emb - ref_emb)
                    if dist < min_dist:
                        min_dist, best_match = dist, name

                threshold = 0.2
                if best_match and min_dist < threshold:
                    label = f"{best_match}"
                    color = (0, 255, 0)

                    if best_match in database and best_match not in recognized_once:
                        info = database[best_match]
                        print(f"\n🎯 Tanınan Kişi: {best_match}")
                        for k, v in info.items():
                            print(f"{k}: {v}")
                        recognized_once.add(best_match)

                elif best_match and min_dist < 0.8:
                    label = "Bilinmiyor"
                    color = (0, 0, 255)
                else:
                    continue  

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Yuz Tanima", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


root = tk.Tk()
root.title("Yüz Tanıma Sistemi")
root.geometry("500x750")
root.configure(bg="#000f28")

try:
    img = Image.open("—Pngtree—technology vector face recognition 25d_5472107.png").resize((400, 350))
    photo = ImageTk.PhotoImage(img)
    tk.Label(root, image=photo, bg="#000f28").pack(pady=10)
except:
    print("Görsel yüklenemedi.")

tk.Label(root, text="Yüz Tanıma Sistemi", font=("Helvetica", 20, "bold"), fg="white", bg="#000f28").pack(pady=10)

def build_button_with_border(master, text, command=None):
    frame = tk.Frame(master, bg="white", highlightthickness=0)
    button = tk.Button(
        frame,
        text=text,
        command=command,
        font=("Helvetica", 14, "bold"),
        bg="#000f28",
        fg="white",
        activebackground="#001f3f",
        activeforeground="white",
        bd=0,
        relief="flat",
        cursor="hand2",
        padx=20,
        pady=10
    )
    button.pack(padx=2, pady=2)  
    return frame

build_button_with_border(root, "Kayıtlı Verilerle Yüz Tanıma", start_face_recognition).pack(pady=10)
build_button_with_border(root, "Yeni Kişi Ekle", add_new_person).pack(pady=10)
build_button_with_border(root, "Çıkış", root.destroy).pack(pady=10)

root.mainloop()
