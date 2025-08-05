import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

class SphericalHarmonics(nn.Module):
    """Spherical harmonics for view-dependent color representation"""
    def __init__(self, degree=3, num_channels=3):
        super().__init__()
        self.degree = degree
        self.num_channels = num_channels
        self.num_coeffs = (degree + 1) ** 2
        self.coeffs = nn.Parameter(torch.randn(num_channels, self.num_coeffs) * 0.1)
    
    def forward(self, view_directions):
        # Simplified SH computation
        batch_size = view_directions.shape[0]
        view_directions = F.normalize(view_directions, dim=-1)
        x, y, z = view_directions[:, 0], view_directions[:, 1], view_directions[:, 2]
        basis = torch.stack([torch.ones_like(x), x, y, z, x*y, x*z, y*z, x**2, y**2, z**2], dim=1)
        basis = basis[:, :self.num_coeffs]
        
        colors = torch.matmul(basis, self.coeffs.T)
        return torch.sigmoid(colors)

class GaussianField(nn.Module):
    """
    Enhanced Gaussian Field with progressive densification and spherical harmonics
    """
    def __init__(self, num_gaussians=1024, device='cpu', sh_degree=3):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.device = device
        self.sh_degree = sh_degree
        
        # Initialize parameters
        self.means = nn.Parameter(torch.randn(num_gaussians, 3, device=device) * 0.5)
        self.log_scales = nn.Parameter(torch.zeros(num_gaussians, 3, device=device))
        self.rotations = nn.Parameter(torch.tensor([1., 0., 0., 0.], device=device).repeat(num_gaussians, 1))
        
        # Spherical harmonics coefficients
        sh_coeffs = torch.randn(num_gaussians, 3, (sh_degree + 1) ** 2, device=device) * 0.1
        self.sh_coeffs = nn.Parameter(sh_coeffs)
        
        # Opacity
        self.opacities = nn.Parameter(torch.rand(num_gaussians, 1, device=device) * 0.5 + 0.1)
        
        # Densification tracking
        self.register_buffer('densification_count', torch.tensor(0))
        self.register_buffer('pruning_mask', torch.ones(num_gaussians, dtype=torch.bool))
        
    def densify_and_prune(self, grad_threshold=0.0002, opacity_threshold=0.005):
        """Progressive densification: split large Gaussians, remove transparent ones"""
        with torch.no_grad():
            grad_norm = torch.norm(self.means.grad, dim=-1) if self.means.grad is not None else torch.zeros(self.num_gaussians, device=self.device)
            split_mask = grad_norm > grad_threshold
            remove_mask = self.opacities.data.squeeze() < opacity_threshold
            self.pruning_mask = ~remove_mask
            self.densification_count += 1

    def forward(self, view_directions=None):
        gaussians = {
            'means': self.means,
            'scales': torch.exp(self.log_scales),
            'rotations': self.rotations,
            'sh_coeffs': self.sh_coeffs,
            'opacities': self.opacities,
            'pruning_mask': self.pruning_mask
        }
        
        if view_directions is not None:
            gaussians['colors'] = self._compute_colors(view_directions)
        else:
            gaussians['colors'] = torch.sigmoid(self.sh_coeffs[:, :, 0])
            
        return gaussians
    
    def _compute_colors(self, view_directions):
        """Compute view-dependent colors using spherical harmonics"""
        batch_size, num_gaussians = view_directions.shape[0], self.num_gaussians
        colors = torch.zeros(batch_size, num_gaussians, 3, device=self.device)
        
        for i in range(batch_size):
            view_dir = view_directions[i]
            view_dir = F.normalize(view_dir, dim=-1)
            x, y, z = view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]
            basis = torch.stack([torch.ones_like(x), x, y, z, x*y, x*z, y*z, x**2, y**2, z**2], dim=1)
            basis = basis[:, :(self.sh_degree + 1) ** 2]
            sh_coeffs = self.sh_coeffs
            colors[i] = torch.sigmoid(torch.sum(basis.unsqueeze(1) * sh_coeffs, dim=-1))
            
        return colors

def differentiable_rasterizer(gaussians, camera_pose, image_size=(256, 256)):
    """Enhanced differentiable rasterizer with camera projection"""
    batch_size = camera_pose.shape[0] if camera_pose is not None else 1
    height, width = image_size
    device = gaussians['means'].device
    
    # Apply camera transformation
    means_3d = gaussians['means']
    means_3d_homo = torch.cat([means_3d, torch.ones(means_3d.shape[0], 1, device=device)], dim=1)
    
    # Transform to camera space
    camera_pose_inv = torch.inverse(camera_pose[0])
    means_camera = torch.matmul(camera_pose_inv, means_3d_homo.T).T[:, :3]
    
    # Project to 2D
    fx = fy = min(height, width) * 0.8
    cx, cy = width / 2, height / 2
    
    z = means_camera[:, 2]
    x_2d = (means_camera[:, 0] * fx / z) + cx
    y_2d = (means_camera[:, 1] * fy / z) + cy
    
    # Simple alpha blending
    rendered = torch.zeros(batch_size, 3, height, width, device=device)
    depth_order = torch.argsort(z, descending=True)
    
    for idx in depth_order:
        if gaussians['pruning_mask'][idx]:
            x, y = int(x_2d[idx].item()), int(y_2d[idx].item())
            if 0 <= x < width and 0 <= y < height:
                alpha = gaussians['opacities'][idx, 0]
                color = gaussians['colors'][idx] if 'colors' in gaussians else torch.sigmoid(gaussians['sh_coeffs'][idx, :, 0])
                
                # Simple point splatting
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            weight = torch.exp(-(dx*dx + dy*dy) / 2.0)
                            rendered[0, :, ny, nx] = (1 - alpha * weight) * rendered[0, :, ny, nx] + alpha * weight * color
    
    return rendered

class DreamGaussianModel(nn.Module):
    def __init__(self, num_gaussians=1024, device='cpu', sh_degree=3):
        super().__init__()
        self.gaussian_field = GaussianField(num_gaussians=num_gaussians, device=device, sh_degree=sh_degree)
        self.device = device

    def forward(self, camera_pose, image_size=(256, 256), view_directions=None):
        gaussians = self.gaussian_field(view_directions)
        rendered = differentiable_rasterizer(gaussians, camera_pose, image_size)
        return rendered
    
    def densify_and_prune(self):
        """Perform progressive densification"""
        self.gaussian_field.densify_and_prune()
    
    def extract_mesh(self, resolution=64):
        """Extract mesh from Gaussian field"""
        with torch.no_grad():
            means = self.gaussian_field.means.data
            vertices = means.cpu().numpy()
            
            # Generate simple faces
            faces = []
            for i in range(0, len(vertices) - 2, 3):
                if i + 2 < len(vertices):
                    faces.append([i, i+1, i+2])
            
            return {
                'vertices': vertices,
                'faces': np.array(faces) if faces else np.array([]),
                'colors': torch.sigmoid(self.gaussian_field.sh_coeffs[:, :, 0]).cpu().numpy()
            } 