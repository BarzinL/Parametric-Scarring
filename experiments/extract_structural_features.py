import torch
import torch.nn.functional as F
from scipy.ndimage import label

def structural_similarity(pattern1, pattern2):
    """
    Measure similarity based on spatial structure, not pixel values.
    """
    # Edge detection (gradient magnitude)
    def get_edges(pattern):
        gx = F.conv2d(pattern.unsqueeze(0).unsqueeze(0), 
                      torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                      dtype=pattern.dtype, device=pattern.device))
        gy = F.conv2d(pattern.unsqueeze(0).unsqueeze(0),
                      torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                      dtype=pattern.dtype, device=pattern.device))
        edges = torch.sqrt(gx**2 + gy**2).squeeze()
        return edges
    
    edges1 = get_edges(pattern1)
    edges2 = get_edges(pattern2)
    
    # Structural similarity on edges only
    edge_sim = torch.cosine_similarity(edges1.flatten(), edges2.flatten(), dim=0)
    
    return edge_sim.item()

# Test on your patterns
from pathlib import Path
import matplotlib.pyplot as plt

pattern_a = plt.imread("../results/experiment_1a_rev1/pattern_a_emergent.png")
pattern_i = plt.imread("../results/experiment_1a_rev1/pattern_i_emergent.png")
pattern_u = plt.imread("../results/experiment_1a_rev1/pattern_u_emergent.png")

# Convert to tensors (assuming grayscale or take single channel)
pattern_a_t = torch.tensor(pattern_a[:,:,0], dtype=torch.float32)
pattern_i_t = torch.tensor(pattern_i[:,:,0], dtype=torch.float32)
pattern_u_t = torch.tensor(pattern_u[:,:,0], dtype=torch.float32)

print("Structural similarities:")
print(f"a-i: {structural_similarity(pattern_a_t, pattern_i_t):.3f}")
print(f"a-u: {structural_similarity(pattern_a_t, pattern_u_t):.3f}")
print(f"i-u: {structural_similarity(pattern_i_t, pattern_u_t):.3f}")