import insightface
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# --- Initialize Buffalo model ---
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)  # CPU

# --- Test one image URL ---
###This is URL for first image.###
###test_url = "http://getbeatmadrid.files.wordpress.com/2013/01/magic-alex.jpg"
test_url = "http://1.bp.blogspot.com/-HNFIL7eKdNs/TxnlvZvisvI/AAAAAAAAANE/AsNempI4Efc/s1600/magicalex.jpg"

try:
    # Download image
    response = requests.get(test_url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_np = np.array(img)

    # Run face detection / get embeddings
    faces = model.get(img_np)
    print(f"Number of faces detected: {len(faces)}")
    if len(faces) > 0:
        print("Embedding vector length:", len(faces[0].embedding))
    else:
        print("No faces detected. Check if the image contains a clear face.")

except Exception as e:
    print(f"Failed to read or process image: {e}")
