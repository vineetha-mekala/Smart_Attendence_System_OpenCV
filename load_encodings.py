# load_encodings.py
import pickle
import numpy as np
import os

DEFAULT_PATH = "encodings.pkl"

def load_encodings(path=DEFAULT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run encodings_generator.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    embeddings = np.array(data["encodings"])
    names = list(data["names"])
    return embeddings, names
