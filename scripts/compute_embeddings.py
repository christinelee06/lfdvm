import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import insightface

# --- CONFIG ---
tsv_file = "/home/christine/lfd/data/ms1m/OpenDataLab___MS-Celeb-1M/raw/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"
embeddings_dir = "/home/christine/lfd/embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

# --- Initialize model ---
print("Initializing InsightFace Buffalo model...")
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)  # CPU
print("Model initialized successfully!")

# --- Read TSV in chunks ---
chunksize = 100  # adjust depending on memory and speed
for chunk in pd.read_csv(tsv_file, sep="\t", header=None, chunksize=chunksize, encoding='utf-8'):
    for idx, row in chunk.iterrows():
        person_id = str(row[0])
        img_url = row[2]  # third column has URLs

        try:
            # Download image
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_np = np.array(img)

            # Get face embeddings
            faces = model.get(img_np)
            if len(faces) == 0:
                print(f"[Row {idx}] No face detected for {person_id}")
                continue

            embedding = faces[0].embedding
            # Save embedding with a clear filename
            save_path = os.path.join(embeddings_dir, f"{person_id}_{idx}.npy")
            np.save(save_path, embedding)
            print(f"[Row {idx}] Embedding saved for {person_id}")

        except Exception as e:
            print(f"[Row {idx}] Skipping {person_id} due to error: {e}")

print("All done! Embeddings saved to:", embeddings_dir)
