import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from model import CNNRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Load Model
model = CNNRNN().to(DEVICE)
model.load_state_dict(torch.load("finalmodel.pth", map_location=DEVICE,weights_only=False)) 
model.eval()


# Inference Transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_batch(image_bytes_list):
    tensors = []
    
    # 1. Process each image
    for img_bytes in image_bytes_list:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        tensor = transform(img) # Shape: [3, 224, 224]
        tensors.append(tensor)
        
    # 2. Stack into a single batch tensor
    # Shape becomes: [Batch_Size, 3, 224, 224]
    batch_tensor = torch.stack(tensors).to(DEVICE)
    
    # 3. Predict all at once (Massive speedup)
    with torch.no_grad():
        batch_preds = model.predict_beam(batch_tensor, beam_size=2)
        
    # 4. Map indices back to words for every image
    results = []
    for preds in batch_preds:
        labels = [VOC_CLASSES[i] for i in preds if i < 20]
        results.append(labels)
        
    return results
