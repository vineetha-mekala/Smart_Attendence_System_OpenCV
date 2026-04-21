# encodings_generator.py
import os
import cv2
import numpy as np
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

STUDENTS_DIR = "students"
OUTPUT_FILE = "encodings.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_encodings():
    # Detector and embedder
    mtcnn = MTCNN(keep_all=False, device=DEVICE)         # detect single best face per image
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    embeddings = []
    names = []

    if not os.path.exists(STUDENTS_DIR):
        print(f"Students folder not found: {STUDENTS_DIR}")
        return

    for student_name in sorted(os.listdir(STUDENTS_DIR)):
        student_folder = os.path.join(STUDENTS_DIR, student_name)
        if not os.path.isdir(student_folder):
            continue

        # Support single image per student or multiple
        for img_name in sorted(os.listdir(student_folder)):
            img_path = os.path.join(student_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue

            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detect and crop face (mtcnn returns PIL crops or tensors)
            try:
                face = mtcnn(img_rgb)
            except Exception as e:
                print(f"mtcnn error for {img_path}: {e}")
                continue

            if face is None:
                print(f"❌ No face detected: {img_path}")
                continue

            # face is a tensor (3,160,160) or batch; ensure correct shape
            if isinstance(face, torch.Tensor):
                face_tensor = face.unsqueeze(0).to(DEVICE) if face.ndim == 3 else face.to(DEVICE)
            else:
                # fallback: convert to tensor via mtcnn already does that normally
                print(f"Unexpected face type for {img_path}, skipping")
                continue

            # get embedding (1,512)
            with torch.no_grad():
                emb = resnet(face_tensor).cpu().numpy().reshape(-1)  # 512-d vector

            embeddings.append(emb)
            names.append(student_name)
            print(f"✔️ Encoded {student_name} / {img_name}")

    # Save
    data = {"encodings": np.vstack(embeddings) if embeddings else np.zeros((0,512)), "names": names}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\n🎉 Saved {len(names)} encodings to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_encodings()
