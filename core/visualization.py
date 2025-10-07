"""
visualization.py - Plotting utilities for patterns and results

Functions for visualizing:
- Substrate state (U, V concentrations)
- Parameter scars (F, K modifications)
- Training curves (similarity over time)
- Experimental summaries
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_state(V_state, filename, title=None, cmap="viridis", vmin=0, vmax=1):
    """
    Save visualization of V concentration.

    Args:
        V_state: 2D tensor or numpy array (substrate.V)
        filename: Output path (str or Path)
        title: Plot title (default: derived from filename)
        cmap: Colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    # Convert to numpy if torch tensor
    if hasattr(V_state, "cpu"):
        V_state = V_state.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(V_state, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="V concentration")

    if title is None:
        title = Path(filename).stem.replace("_", " ").title()
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_parameter_scars(substrate, filename):
    """
    Visualize F and K parameter changes (scars).

    Args:
        substrate: GrayScottSubstrate instance
        filename: Output path
    """
    f_change = (substrate.F - substrate.default_f).cpu().numpy()
    k_change = (substrate.K - substrate.default_k).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # F changes (red = increased feed, blue = decreased)
    im1 = axes[0].imshow(f_change, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
    axes[0].set_title("F Parameter Changes (Feed Rate)")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], label="ΔF")

    # K changes (red = decreased kill, blue = increased)
    im2 = axes[1].imshow(k_change, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
    axes[1].set_title("K Parameter Changes (Kill Rate)")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], label="ΔK")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_training_curves(similarities_dict, filename, title="Training Convergence"):
    """
    Plot similarity over time for multiple conditions.

    Args:
        similarities_dict: Dict mapping labels to similarity lists
            e.g., {'Scarred': [0.1, 0.3, 0.5, ...], 'Control': [...]}
        filename: Output path
        title: Plot title
    """
    plt.figure(figsize=(8, 6))

    for label, similarities in similarities_dict.items():
        plt.plot(similarities, label=label, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Similarity to Target")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_recall_curves(recalls_dict, filename, title="Recall Quality"):
    """
    Plot recall quality over time.

    Args:
        recalls_dict: Dict mapping labels to recall quality lists
        filename: Output path
        title: Plot title
    """
    plt.figure(figsize=(8, 6))

    for label, recalls in recalls_dict.items():
        plt.plot(recalls, label=label, linewidth=2)

    plt.xlabel("Timestep")
    plt.ylabel("Similarity to Target")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_multi_pattern_recall(
    recall_qualities, filename, title="Multi-Pattern Capacity"
):
    """
    Bar plot showing recall quality for multiple patterns.

    Args:
        recall_qualities: Dict mapping pattern names to recall quality
            e.g., {'Pattern A': 0.85, 'Pattern B': 0.92, ...}
        filename: Output path
        title: Plot title
    """
    plt.figure(figsize=(8, 6))

    names = list(recall_qualities.keys())
    values = list(recall_qualities.values())

    plt.bar(names, values)
    plt.ylabel("Recall Quality")
    plt.title(title)
    plt.axhline(y=0.7, color="g", linestyle="--", label="Success Threshold (0.7)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_experiment_summary(results_dict, output_dir):
    """
    Generate comprehensive summary figure with 4 subplots.

    Args:
        results_dict: Dict with keys:
            - 'training_scarred': list
            - 'training_control': list
            - 'scar_accumulation': list
            - 'recall_scarred': list
            - 'recall_control': list
            - 'multi_pattern_qualities': dict
        output_dir: Directory to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Training convergence
    axes[0, 0].plot(
        results_dict["training_scarred"], label="With Scarring", linewidth=2
    )
    axes[0, 0].plot(
        results_dict["training_control"], label="Control (No Scars)", linewidth=2
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Similarity to Target")
    axes[0, 0].set_title("Phase 1: Training Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Scar accumulation
    axes[0, 1].plot(results_dict["scar_accumulation"], color="red", linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Total Scar Strength")
    axes[0, 1].set_title("Scar Accumulation During Training")
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Recall quality
    axes[1, 0].plot(results_dict["recall_scarred"], label="Scarred System", linewidth=2)
    axes[1, 0].plot(
        results_dict["recall_control"], label="Baseline System", linewidth=2
    )
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Similarity to Target")
    axes[1, 0].set_title("Phase 2: Recall Quality")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Multi-pattern capacity
    qualities = results_dict["multi_pattern_qualities"]
    axes[1, 1].bar(qualities.keys(), qualities.values())
    axes[1, 1].set_ylabel("Final Recall Quality")
    axes[1, 1].set_title("Phase 3: Multi-Pattern Capacity")
    axes[1, 1].axhline(y=0.7, color="g", linestyle="--", label="Success Threshold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_dir) / "summary_results.png"
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_interference_matrix(interference_matrix, pattern_names, filename):
    """
    Visualize pattern cross-activation as heatmap.

    Args:
        interference_matrix: N×N numpy array
        pattern_names: List of pattern names
        filename: Output path
    """
    plt.figure(figsize=(8, 7))

    im = plt.imshow(interference_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, label="Activation Similarity")

    # Set tick labels
    plt.xticks(range(len(pattern_names)), pattern_names, rotation=45)
    plt.yticks(range(len(pattern_names)), pattern_names)

    # Add text annotations
    for i in range(len(pattern_names)):
        for j in range(len(pattern_names)):
            text = plt.text(
                j,
                i,
                f"{interference_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.xlabel("Activated Pattern")
    plt.ylabel("Presented Pattern")
    plt.title("Pattern Interference Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def create_animation_frames(
    substrate, target_pattern, num_steps, output_dir, prefix="frame"
):
    """
    Save individual frames for creating animation.

    Args:
        substrate: GrayScottSubstrate instance
        target_pattern: Target pattern (for comparison)
        num_steps: Number of simulation steps
        output_dir: Directory to save frames
        prefix: Filename prefix

    Returns:
        list: Paths to saved frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    frame_paths = []

    for step in range(num_steps):
        substrate.simulate_step()

        # Save every N steps (to reduce file count)
        if step % 10 == 0:
            filename = output_dir / f"{prefix}_{step:04d}.png"
            save_state(substrate.V, filename, title=f"Step {step}")
            frame_paths.append(filename)

    return frame_paths
