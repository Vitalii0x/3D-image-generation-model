import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import CLIPVoxelModel

# Dummy dataset: random text prompts and random voxel targets
def get_dummy_batch(batch_size, voxel_dim, device):
    texts = ["a cube", "a sphere", "a pyramid", "a chair"] * (batch_size // 4 + 1)
    texts = texts[:batch_size]
    targets = torch.rand(batch_size, 1, voxel_dim, voxel_dim, voxel_dim, device=device)
    return texts, targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voxel_dim = 32
    model = CLIPVoxelModel(device=device, voxel_dim=voxel_dim).to(device)
    optimizer = optim.Adam(model.generator.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    epochs = 5
    batch_size = 4

    for epoch in range(epochs):
        texts, targets = get_dummy_batch(batch_size, voxel_dim, device)
        optimizer.zero_grad()
        outputs = model(text=texts[0])  # Only use first text for demo
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main() 