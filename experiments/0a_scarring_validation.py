import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("=== Scarring Validation Experiment ===\n")

# Configuration
WIDTH, HEIGHT = 256, 256  # Smaller for faster computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Physics parameters
DEFAULT_F = 0.037
DEFAULT_K = 0.060
Du, Dv, dt = 0.16, 0.08, 1.0
DECAY_RATE = 0.9995

# Scarring parameters
SCAR_STRENGTH = 0.003
SIMILARITY_THRESHOLD = 0.5  # Apply scarring when pattern matches this well

# Create output directory
output_dir = Path("scarring_results")
output_dir.mkdir(exist_ok=True)

# Initialize tensors
F = torch.full((HEIGHT, WIDTH), DEFAULT_F, device=device)
K = torch.full((HEIGHT, WIDTH), DEFAULT_K, device=device)
U = torch.ones(HEIGHT, WIDTH, device=device)
V = torch.zeros(HEIGHT, WIDTH, device=device)

# Create coordinate grids
y_grid, x_grid = torch.meshgrid(
    torch.arange(HEIGHT, device=device),
    torch.arange(WIDTH, device=device),
    indexing="ij",
)

# Laplacian kernel
laplacian_kernel = torch.tensor(
    [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]], device=device
).reshape(1, 1, 3, 3)


def create_target_pattern(pattern_type="square_spots", region_offset=(0, 0)):
    """Generate target patterns for training"""
    target = torch.zeros(HEIGHT, WIDTH, device=device)
    offset_x, offset_y = region_offset

    if pattern_type == "square_spots":
        # Four spots in square arrangement
        positions = [
            (64 + offset_x, 64 + offset_y),
            (64 + offset_x, 192 + offset_y),
            (192 + offset_x, 64 + offset_y),
            (192 + offset_x, 192 + offset_y),
        ]
        for x, y in positions:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 < 15**2
                target[mask] = 0.8

    elif pattern_type == "ring":
        # Ring pattern
        center_x, center_y = 128 + offset_x, 128 + offset_y
        if 0 <= center_x < WIDTH and 0 <= center_y < HEIGHT:
            dist = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
            ring_mask = (dist > 40) & (dist < 60)
            target[ring_mask] = 0.8

    elif pattern_type == "line":
        # Diagonal line
        for i in range(50, 200):
            x, y = i + offset_x, i + offset_y
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 < 8**2
                target[mask] = 0.8

    return target


def simulate_step():
    """Single reaction-diffusion timestep"""
    global U, V

    U_padded = torch.nn.functional.pad(
        U.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular"
    )
    V_padded = torch.nn.functional.pad(
        V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular"
    )

    lap_U = torch.nn.functional.conv2d(U_padded, laplacian_kernel).squeeze()
    lap_V = torch.nn.functional.conv2d(V_padded, laplacian_kernel).squeeze()

    uvv = U * V * V
    U = U + (Du * lap_U - uvv + F * (1 - U)) * dt
    V = V + (Dv * lap_V + uvv - (K + F) * V) * dt

    V *= DECAY_RATE
    U.clamp_(0, 1)
    V.clamp_(0, 1)


def pattern_similarity(V_state, target):
    """Cosine similarity between current state and target"""
    v_flat = V_state.flatten()
    t_flat = target.flatten()
    if v_flat.sum() < 0.01 or t_flat.sum() < 0.01:
        return 0.0
    return torch.cosine_similarity(v_flat, t_flat, dim=0).item()


def apply_scarring(target, strength=SCAR_STRENGTH):
    """Modify F/K where pattern matches target"""
    global F, K

    # Create match map: where both V and target are active
    match_map = (V > 0.3) & (target > 0.3)

    # Strengthen F where pattern exists
    F[match_map] += strength
    # Reduce K to stabilize pattern
    K[match_map] -= strength * 0.5

    # Keep within valid ranges
    F.clamp_(0.01, 0.08)
    K.clamp_(0.03, 0.08)


def scar_strength():
    """Measure total divergence from baseline"""
    f_div = torch.abs(F - DEFAULT_F).sum().item()
    k_div = torch.abs(K - DEFAULT_K).sum().item()
    return f_div + k_div


def train_pattern(target, num_iterations=1000, apply_scars=True):
    """Train the system to produce a target pattern"""
    global U, V

    print(f"Training {'WITH' if apply_scars else 'WITHOUT'} scarring...")

    # Initialize with weak version of target plus noise
    V[:] = target * 0.2 + torch.randn_like(target) * 0.1
    V.clamp_(0, 1)
    U[:] = 1.0

    similarities = []
    scar_values = []

    for i in range(num_iterations):
        simulate_step()

        sim = pattern_similarity(V, target)
        similarities.append(sim)
        scar_values.append(scar_strength())

        # Apply scarring when pattern emerges well
        if apply_scars and sim > SIMILARITY_THRESHOLD:
            apply_scarring(target)

        if i % 200 == 0:
            print(
                f"  Iteration {i:4d}: Similarity = {sim:.3f}, Scars = {scar_values[-1]:.1f}"
            )

    final_sim = similarities[-1]
    print(f"  Final similarity: {final_sim:.3f}\n")

    return similarities, scar_values


def test_recall(target, perturbation_strength=0.15, num_steps=500):
    """Test if weak perturbation reconstructs the pattern"""
    global U, V

    print(f"Testing recall with {perturbation_strength * 100:.0f}% perturbation...")

    # Reset to weak perturbation of target
    V[:] = target * perturbation_strength
    V += torch.randn_like(V) * 0.05
    V.clamp_(0, 1)
    U[:] = 1.0

    similarities = []

    for i in range(num_steps):
        simulate_step()
        sim = pattern_similarity(V, target)
        similarities.append(sim)

        if i % 100 == 0:
            print(f"  Step {i:3d}: Similarity = {sim:.3f}")

    final_sim = similarities[-1]
    print(f"  Final recall quality: {final_sim:.3f}\n")

    return similarities


def save_state(V_state, filename):
    """Save visualization of V state"""
    plt.figure(figsize=(6, 6))
    plt.imshow(V_state.cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="V intensity")
    plt.title(filename.stem.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def run_experiment():
    """Main experimental protocol"""
    global F, K, U, V

    results = {}

    # ===== PHASE 1: Single Pattern Training =====
    print("=" * 50)
    print("PHASE 1: Single Pattern Training")
    print("=" * 50 + "\n")

    target = create_target_pattern("square_spots")
    save_state(target, output_dir / "target_pattern.png")

    # Train WITH scarring
    F[:] = DEFAULT_F
    K[:] = DEFAULT_K
    train_sim_scarred, train_scars = train_pattern(
        target, num_iterations=1000, apply_scars=True
    )
    V_trained = V.clone()
    save_state(V_trained, output_dir / "after_training_scarred.png")
    F_scarred = F.clone()
    K_scarred = K.clone()

    # Train WITHOUT scarring (control)
    F[:] = DEFAULT_F
    K[:] = DEFAULT_K
    train_sim_control, _ = train_pattern(target, num_iterations=1000, apply_scars=False)
    V_control = V.clone()
    save_state(V_control, output_dir / "after_training_control.png")

    results["phase1"] = {
        "scarred_final_similarity": train_sim_scarred[-1],
        "control_final_similarity": train_sim_control[-1],
        "total_scar_strength": train_scars[-1],
    }

    # ===== PHASE 2: Pattern Recall =====
    print("=" * 50)
    print("PHASE 2: Pattern Recall Test")
    print("=" * 50 + "\n")

    # Test recall with scarred parameters
    F[:] = F_scarred
    K[:] = K_scarred
    recall_sim_scarred = test_recall(target, perturbation_strength=0.15)
    V_recall_scarred = V.clone()
    save_state(V_recall_scarred, output_dir / "recall_scarred.png")

    # Test recall with baseline parameters (control)
    F[:] = DEFAULT_F
    K[:] = DEFAULT_K
    recall_sim_control = test_recall(target, perturbation_strength=0.15)
    V_recall_control = V.clone()
    save_state(V_recall_control, output_dir / "recall_control.png")

    results["phase2"] = {
        "scarred_recall_quality": recall_sim_scarred[-1],
        "control_recall_quality": recall_sim_control[-1],
        "improvement_ratio": recall_sim_scarred[-1] / max(recall_sim_control[-1], 0.01),
    }

    # ===== PHASE 3: Multiple Patterns =====
    print("=" * 50)
    print("PHASE 3: Multiple Pattern Capacity")
    print("=" * 50 + "\n")

    F[:] = DEFAULT_F
    K[:] = DEFAULT_K

    patterns = {
        "pattern_A": create_target_pattern("square_spots", region_offset=(-64, -64)),
        "pattern_B": create_target_pattern("ring", region_offset=(0, 0)),
        "pattern_C": create_target_pattern("line", region_offset=(0, 0)),
    }

    # Train all three patterns sequentially
    for name, pattern in patterns.items():
        print(f"\nTraining {name}...")
        train_pattern(pattern, num_iterations=500, apply_scars=True)

    # Test recall for each
    recall_qualities = {}
    for name, pattern in patterns.items():
        print(f"\nRecalling {name}...")
        recall_sim = test_recall(pattern, perturbation_strength=0.2, num_steps=300)
        recall_qualities[name] = recall_sim[-1]
        save_state(V, output_dir / f"recall_{name}.png")

    results["phase3"] = {
        "pattern_recall_qualities": recall_qualities,
        "average_recall": np.mean(list(recall_qualities.values())),
        "min_recall": min(recall_qualities.values()),
    }

    # ===== Generate Summary Plots =====
    print("\n" + "=" * 50)
    print("Generating summary plots...")
    print("=" * 50 + "\n")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(train_sim_scarred, label="With Scarring", linewidth=2)
    axes[0, 0].plot(train_sim_control, label="Control (No Scars)", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Similarity to Target")
    axes[0, 0].set_title("Phase 1: Training Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_scars, color="red", linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Total Scar Strength")
    axes[0, 1].set_title("Scar Accumulation During Training")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(recall_sim_scarred, label="Scarred System", linewidth=2)
    axes[1, 0].plot(recall_sim_control, label="Baseline System", linewidth=2)
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Similarity to Target")
    axes[1, 0].set_title("Phase 2: Recall Quality")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(recall_qualities.keys(), recall_qualities.values())
    axes[1, 1].set_ylabel("Final Recall Quality")
    axes[1, 1].set_title("Phase 3: Multi-Pattern Capacity")
    axes[1, 1].axhline(y=0.7, color="g", linestyle="--", label="Success Threshold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "summary_results.png", dpi=150)
    plt.close()

    # Visualize F/K parameter changes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    f_change = (F - DEFAULT_F).cpu().numpy()
    k_change = (K - DEFAULT_K).cpu().numpy()

    im1 = axes[0].imshow(f_change, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
    axes[0].set_title("F Parameter Changes (Scars)")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(k_change, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
    axes[1].set_title("K Parameter Changes (Scars)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_scars.png", dpi=150)
    plt.close()

    # Save results to JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ===== Final Report =====
    print("\n" + "=" * 50)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 50 + "\n")

    print("PHASE 1: Single Pattern Training")
    print(
        f"  Scarred system final similarity: {results['phase1']['scarred_final_similarity']:.3f}"
    )
    print(
        f"  Control system final similarity: {results['phase1']['control_final_similarity']:.3f}"
    )
    print(f"  Total scar strength: {results['phase1']['total_scar_strength']:.1f}")

    print("\nPHASE 2: Pattern Recall")
    print(
        f"  Scarred recall quality: {results['phase2']['scarred_recall_quality']:.3f}"
    )
    print(
        f"  Control recall quality: {results['phase2']['control_recall_quality']:.3f}"
    )
    print(f"  Improvement ratio: {results['phase2']['improvement_ratio']:.2f}x")

    print("\nPHASE 3: Multiple Pattern Capacity")
    for name, quality in results["phase3"]["pattern_recall_qualities"].items():
        print(f"  {name}: {quality:.3f}")
    print(f"  Average recall: {results['phase3']['average_recall']:.3f}")
    print(f"  Minimum recall: {results['phase3']['min_recall']:.3f}")

    print("\n" + "=" * 50)
    print("INTERPRETATION")
    print("=" * 50 + "\n")

    # Success criteria evaluation
    success_threshold = 0.7
    improvement_threshold = 1.5

    if results["phase2"]["scarred_recall_quality"] > success_threshold:
        print("✓ SUCCESS: Scarred system shows strong recall (>0.7)")
    else:
        print("✗ PARTIAL: Scarred recall below success threshold")

    if results["phase2"]["improvement_ratio"] > improvement_threshold:
        print(
            f"✓ SUCCESS: {results['phase2']['improvement_ratio']:.1f}x improvement over baseline"
        )
    else:
        print("✗ PARTIAL: Scarring doesn't provide substantial advantage")

    if results["phase3"]["min_recall"] > 0.5:
        print("✓ SUCCESS: Multiple patterns can coexist")
    else:
        print("✗ FAILURE: Pattern interference too high")

    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    run_experiment()
