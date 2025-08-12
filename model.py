import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional, Union, List

class VoxelGenerator(nn.Module):
    def __init__(self, latent_dim=512, voxel_dim=32, hidden_dims=[1024, 2048, 4096], dropout=0.1):
        super().__init__()
        self.voxel_dim = voxel_dim
        self.latent_dim = latent_dim
        
        # Build a more sophisticated network
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer to output voxel grid
        layers.append(nn.Linear(prev_dim, voxel_dim ** 3))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z, temperature=1.0):
        """
        Generate voxel grid from latent representation
        Args:
            z: Latent representation [batch_size, latent_dim]
            temperature: Temperature for sigmoid scaling (higher = sharper)
        """
        voxels = self.network(z)
        # Apply temperature scaling to sigmoid for better control
        voxels = torch.sigmoid(voxels / temperature)
        return voxels.view(-1, 1, self.voxel_dim, self.voxel_dim, self.voxel_dim)

class CLIPVoxelModel(nn.Module):
    def __init__(self, device='cpu', voxel_dim=32, clip_model_name="ViT-B/32", dropout=0.1):
        super().__init__()
        self.device = device
        self.voxel_dim = voxel_dim
        
        # Load CLIP model
        try:
            self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
            self.clip_model.eval()  # Set to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
        
        # Get CLIP feature dimension
        self.clip_dim = self.clip_model.text_projection.shape[1]
        
        # Initialize generator with CLIP dimension
        self.generator = VoxelGenerator(
            latent_dim=self.clip_dim, 
            voxel_dim=voxel_dim, 
            dropout=dropout
        ).to(device)

    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text using CLIP
        Args:
            text: Single text string or list of text strings
        Returns:
            Text features tensor [batch_size, clip_dim]
        """
        if isinstance(text, str):
            text = [text]
        
        try:
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                # Normalize features
                text_features = F.normalize(text_features, p=2, dim=1)
            return text_features
        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image using CLIP
        Args:
            image: Image tensor [batch_size, channels, height, width]
        Returns:
            Image features tensor [batch_size, clip_dim]
        """
        try:
            image = image.to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
            return image_features
        except Exception as e:
            raise RuntimeError(f"Image encoding failed: {e}")

    def forward(self, text: Optional[Union[str, List[str]]] = None, 
                image: Optional[torch.Tensor] = None, 
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate voxel grid from text or image
        Args:
            text: Text prompt(s) for generation
            image: Image tensor for generation
            temperature: Temperature for voxel generation (higher = sharper)
        Returns:
            Voxel grid tensor [batch_size, 1, voxel_dim, voxel_dim, voxel_dim]
        """
        if text is not None and image is not None:
            raise ValueError("Please provide either text OR image, not both.")
        
        if text is None and image is None:
            raise ValueError("Either text or image must be provided.")
        
        # Encode input
        if text is not None:
            z = self.encode_text(text)
        else:
            z = self.encode_image(image)
        
        # Generate voxels
        voxels = self.generator(z, temperature=temperature)
        return voxels
    
    def generate_from_text(self, text: Union[str, List[str]], temperature: float = 1.0) -> torch.Tensor:
        """Convenience method for text-to-voxel generation"""
        return self.forward(text=text, temperature=temperature)
    
    def generate_from_image(self, image: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Convenience method for image-to-voxel generation"""
        return self.forward(image=image, temperature=temperature)
    
    def get_voxel_statistics(self, voxels: torch.Tensor) -> dict:
        """
        Get statistics about generated voxels
        Args:
            voxels: Voxel tensor [batch_size, 1, voxel_dim, voxel_dim, voxel_dim]
        Returns:
            Dictionary with voxel statistics
        """
        with torch.no_grad():
            stats = {
                'mean': voxels.mean().item(),
                'std': voxels.std().item(),
                'min': voxels.min().item(),
                'max': voxels.max().item(),
                'occupancy': (voxels > 0.5).float().mean().item(),
                'sparsity': (voxels < 0.1).float().mean().item()
            }
        return stats 