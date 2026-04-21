# system_process.py
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from load_encodings import load_encodings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = "processed_images"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# create detector & embedder once
_mtcnn = MTCNN(keep_all=True, device=DEVICE)   # keep_all to detect all faces in class photo
_resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# load known encodings
KNOWN_EMBS, KNOWN_NAMES = load_encodings()  # KNOWN_EMBS shape (N,512)
if KNOWN_EMBS.size == 0:
    KNOWN_EMBS = np.zeros((0,512))

def l2_normalize(x):
    # normalize vectors
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

def recognize_faces_in_image(img_path, save_name=None, threshold=0.8):
    """
    Returns: recognized_names (list), unknown_boxes (list of boxes), annotated_image (BGR np), save_path
    threshold: cosine similarity threshold (0..1). Higher -> stricter (0.8 is typical)
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # detect faces (list of PIL/cropped tensors depending on facenet-pytorch)
    boxes, probs = _mtcnn.detect(img_rgb)  # boxes: Nx4 array or None

    recognized = []
    unknowns = []
    annotated = img_bgr.copy()

    if boxes is None:
        # nothing detected, save and return
        save_path = os.path.join(PROCESSED_DIR, save_name or os.path.basename(img_path))
        cv2.imwrite(save_path, annotated)
        return recognized, unknowns, annotated, save_path

    # Extract faces by crops using mtcnn
    faces = _mtcnn.extract(img_rgb, boxes, None)  # returns list of tensors (N,3,160,160) or a tensor
    if isinstance(faces, torch.Tensor):
        faces_t = faces.to(DEVICE)
    else:
        faces_t = torch.stack(faces).to(DEVICE)

    with torch.no_grad():
        embs = _resnet(faces_t).cpu().numpy()  # (num_faces,512)

    # normalize known embeddings
    if KNOWN_EMBS.shape[0] > 0:
        known_norm = KNOWN_EMBS / np.linalg.norm(KNOWN_EMBS, axis=1, keepdims=True)
    else:
        known_norm = np.zeros((0,512))

    for i, box in enumerate(boxes):
        emb = embs[i]
        emb_norm = emb / np.linalg.norm(emb)

        name = "Unknown"
        best_sim = -1.0

        if known_norm.shape[0] > 0:
            sims = known_norm.dot(emb_norm)  # cosine similarities
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= threshold:
                name = KNOWN_NAMES[best_idx]
        # draw
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, f"{name} {best_sim:.2f}", (x1, max(20,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if name != "Unknown":
            recognized.append(name)
        else:
            unknowns.append((x1,y1,x2,y2))

    save_path = os.path.join(PROCESSED_DIR, save_name or os.path.basename(img_path))
    cv2.imwrite(save_path, annotated)

    return recognized, unknowns, annotated, save_path
