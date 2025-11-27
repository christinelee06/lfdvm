import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import insightface
import random

# --- CONFIG ---
TSV_FILE = "/home/christine/lfd/data/ms1m/OpenDataLab___MS-Celeb-1M/raw/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"
EMB_DIR = "/home/christine/lfd/embeddings/antelope"
os.makedirs(EMB_DIR, exist_ok=True)

SAMPLE_SIZE = 5000  # CPU-friendly sample
CHUNKSIZE = 100

# --- Initialize model ---
print("Initializing InsightFace Buffalo model...")
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)
print("Model initialized successfully!")

# --- Random sample rows from TSV ---
total_rows = sum(1 for _ in open(TSV_FILE)) - 1
sample_rows = sorted(random.sample(range(total_rows), SAMPLE_SIZE))

for chunk in pd.read_csv(TSV_FILE, sep="\t", header=None, chunksize=CHUNKSIZE):
    for idx, row in chunk.iterrows():
        if idx not in sample_rows:
            continue
        person_id = str(row[0])
        img_url = row[2]

        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_np = np.array(img)

            faces = model.get(img_np)
            if len(faces) == 0:
                print(f"[Row {idx}] No face detected for {person_id}")
                continue

            embedding = faces[0].embedding
            save_path = os.path.join(EMB_DIR, f"{person_id}_{idx}.npy")
            np.save(save_path, embedding)
            print(f"[Row {idx}] Saved embedding for {person_id}")

        except Exception as e:
            print(f"[Row {idx}] Skipping {person_id} due to error: {e}")

print("All done! Embeddings saved to:", EMB_DIR)
