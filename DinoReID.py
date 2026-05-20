import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

model_dims = {
    "dinov2_vits14": 384
}

class DinoReID:
    def __init__(self, model_name="dinov2_vits14", device="cuda"):
        self.device = device
        self.model_dims = model_dims[model_name]
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def __call__(self, crops):
        # crops: list of bgr np arrays
        if len(crops) == 0:
            return np.empty((0, self.model_dims), dtype=np.float32)
        
        tensors = []
        for c in crops:
            if c.size == 0:
                c = np.zeros((224, 224, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.transform(pil))
        
        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy().astype(np.float32)