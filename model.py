import torch
import torch.nn as nn
import clip

class VoxelGenerator(nn.Module):
    def __init__(self, latent_dim=512, voxel_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, voxel_dim ** 3),
            nn.Sigmoid()  # Output in [0,1] for occupancy
        )
        self.voxel_dim = voxel_dim

    def forward(self, z):
        voxels = self.fc(z)
        return voxels.view(-1, 1, self.voxel_dim, self.voxel_dim, self.voxel_dim)

class CLIPVoxelModel(nn.Module):
    def __init__(self, device='cpu', voxel_dim=32):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.generator = VoxelGenerator(latent_dim=512, voxel_dim=voxel_dim)
        self.device = device

    def encode_text(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        return text_features

    def encode_image(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def forward(self, text=None, image=None):
        if text is not None:
            z = self.encode_text(text)
        elif image is not None:
            z = self.encode_image(image)
        else:
            raise ValueError("Either text or image must be provided.")
        voxels = self.generator(z)
        return voxels 