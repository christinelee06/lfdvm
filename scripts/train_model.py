import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO

# --- PATHS ---
TSV_PATH = "/home/christine/lfd/data/ms1m/OpenDataLab___MS-Celeb-1M/raw/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# --- CUSTOM DATASET ---
class MS1MDataset(Dataset):
    def __init__(self, tsv_path, transform=None):
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        person_id = int(row['identity_id'])  # adjust column name
        img_b64 = row['image']  # adjust column name
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, person_id

# --- LOAD DATA ---
dataset = MS1MDataset(TSV_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# --- SIMPLE TRAIN LOOP EXAMPLE ---
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: ResNet18 as feature extractor
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10575)  # replace with your # of identities
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- TRAIN LOOP ---
for epoch in range(1):  # increase epochs
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done. Loss: {loss.item():.4f}")

# --- SAVE MODEL ---
torch.save(model.state_dict(), "/home/christine/lfd/models/ms1m_model.pth")
print("Model saved to /home/christine/lfd/models/ms1m_model.pth")
