"""
patterns.py - Pattern generators for substrate training

Two types of patterns:
1. Geometric: Hand-designed shapes (squares, rings, lines)
2. Sensory: Real-world data (audio spectrograms, images)

The substrate doesn't care what the pattern represents - it's all just
2D arrays of values. But for humans, different patterns test different
capabilities.
"""

import torch
import numpy as np


def create_geometric_pattern(pattern_type, width, height, device, region_offset=(0, 0)):
    """
    Generate hand-designed geometric patterns.

    These are "toy" patterns for validation experiments.

    Args:
        pattern_type: 'square_spots', 'ring', 'line', 'cross', 'diagonal_stripes'
        width: Pattern width
        height: Pattern height
        device: torch.device
        region_offset: (x_offset, y_offset) for position variation

    Returns:
        torch.Tensor: 2D pattern (values in [0, 1])
    """
    pattern = torch.zeros(height, width, device=device)
    offset_x, offset_y = region_offset

    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    if pattern_type == "square_spots":
        # Four spots in square arrangement
        positions = [
            (64 + offset_x, 64 + offset_y),
            (64 + offset_x, 192 + offset_y),
            (192 + offset_x, 64 + offset_y),
            (192 + offset_x, 192 + offset_y),
        ]
        for x, y in positions:
            if 0 <= x < width and 0 <= y < height:
                mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 < 15**2
                pattern[mask] = 0.8

    elif pattern_type == "ring":
        # Ring pattern (annulus)
        center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
        if 0 <= center_x < width and 0 <= center_y < height:
            dist = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
            ring_mask = (dist > 40) & (dist < 60)
            pattern[ring_mask] = 0.8

    elif pattern_type == "line":
        # Diagonal line
        for i in range(50, 200):
            x, y = i + offset_x, i + offset_y
            if 0 <= x < width and 0 <= y < height:
                mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 < 8**2
                pattern[mask] = 0.8

    elif pattern_type == "cross":
        # Cross pattern (+ shape)
        center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
        # Horizontal bar
        h_mask = (torch.abs(y_grid - center_y) < 10) & (
            torch.abs(x_grid - center_x) < 60
        )
        # Vertical bar
        v_mask = (torch.abs(x_grid - center_x) < 10) & (
            torch.abs(y_grid - center_y) < 60
        )
        pattern[h_mask | v_mask] = 0.8

    elif pattern_type == "diagonal_stripes":
        # Diagonal stripes (for testing interference)
        stripe_width = 20
        for i in range(-height, width, stripe_width * 2):
            for dx in range(stripe_width):
                x_line = i + dx + offset_x
                mask = (x_grid - y_grid) == x_line
                pattern[mask] = 0.8

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return pattern


def create_perturbed_pattern(base_pattern, strength=0.15, noise_std=0.05):
    """
    Create weak + noisy version of pattern (for recall testing).

    Args:
        base_pattern: Original pattern
        strength: Scaling factor (0-1)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        torch.Tensor: Perturbed pattern
    """
    perturbed = base_pattern * strength
    perturbed += torch.randn_like(perturbed) * noise_std
    perturbed.clamp_(0, 1)
    return perturbed


def create_composite_pattern(patterns_list, weights=None):
    """
    Combine multiple patterns (for testing composition).

    Args:
        patterns_list: List of patterns to combine
        weights: Optional list of weights (default: equal weighting)

    Returns:
        torch.Tensor: Combined pattern
    """
    if weights is None:
        weights = [1.0 / len(patterns_list)] * len(patterns_list)

    composite = torch.zeros_like(patterns_list[0])
    for pattern, weight in zip(patterns_list, weights):
        composite += pattern * weight

    composite.clamp_(0, 1)
    return composite


# ============================================================================
# AUDIO/PHONEME PATTERN GENERATION (Experiment 1A)
# ============================================================================


def synthesize_vowel_phoneme(phoneme, duration=0.5, sample_rate=22050):
    """
    Synthesize a vowel phoneme using formant frequencies.

    Formants are the resonant frequencies of the vocal tract.
    Different vowels have characteristic formant patterns.

    Args:
        phoneme: One of 'a', 'i', 'u'
        duration: Length of sound in seconds
        sample_rate: Audio sample rate in Hz

    Returns:
        numpy array: Audio signal (1D)
    """
    # Formant frequencies (F1, F2) for vowels
    # Source: Peterson & Barney (1952) - classic phonetics study
    formants = {
        "a": (700, 1200),  # /a/ as in "father"
        "i": (300, 2300),  # /i/ as in "see"
        "u": (300, 900),  # /u/ as in "boot"
    }

    if phoneme not in formants:
        raise ValueError(f"Unknown phoneme: {phoneme}. Use 'a', 'i', or 'u'")

    f1, f2 = formants[phoneme]

    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Synthesize as sum of two sine waves (fundamental + second formant)
    # F1 is stronger (amplitude 1.0), F2 is weaker (amplitude 0.5)
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    # Add slight amplitude envelope (fade in/out to avoid clicks)
    envelope = np.hanning(len(signal))
    signal = signal * envelope

    # Normalize to [-1, 1]
    signal = signal / np.max(np.abs(signal))

    return signal


def audio_to_spectrogram(audio_signal, sample_rate=22050, n_fft=512, hop_length=256):
    """
    Convert audio signal to spectrogram using Short-Time Fourier Transform (STFT).

    Args:
        audio_signal: 1D numpy array of audio samples
        sample_rate: Sample rate in Hz
        n_fft: FFT window size (frequency resolution)
        hop_length: Hop between windows (time resolution)

    Returns:
        numpy array: 2D spectrogram (frequency × time), magnitudes only
    """
    from scipy import signal as sp_signal

    # Compute STFT
    frequencies, times, stft = sp_signal.stft(
        audio_signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    # Get magnitude (discard phase)
    magnitude = np.abs(stft)

    # Convert to log scale (dB) for better visualization
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    return magnitude_db


def spectrogram_to_pattern(spectrogram, target_size=(256, 256), device="cpu"):
    """
    Convert spectrogram to substrate-compatible pattern.

    Args:
        spectrogram: 2D numpy array or tensor (frequency × time)
        target_size: Desired dimensions
        device: torch.device

    Returns:
        torch.Tensor: Pattern suitable for substrate
    """
    # Convert to torch tensor if numpy
    if isinstance(spectrogram, np.ndarray):
        pattern = torch.from_numpy(spectrogram).float()
    else:
        pattern = spectrogram.float()

    # Resize if needed
    if pattern.shape != target_size:
        pattern = torch.nn.functional.interpolate(
            pattern.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    # Normalize to [0, 1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)

    return pattern.to(device)


def create_phoneme_pattern(phoneme, target_size=(256, 256), device="cpu", duration=0.5):
    """
    Generate a phoneme and convert to substrate-compatible pattern.

    This is the main entry point for Experiment 1A.

    Args:
        phoneme: One of 'a', 'i', 'u'
        target_size: Desired pattern dimensions
        device: torch.device
        duration: Phoneme duration in seconds

    Returns:
        tuple: (pattern tensor, audio signal, sample_rate)
    """
    # Synthesize audio
    audio_signal = synthesize_vowel_phoneme(phoneme, duration)
    sample_rate = 22050

    # Convert to spectrogram
    spectrogram = audio_to_spectrogram(audio_signal, sample_rate)

    # Convert to pattern
    pattern = spectrogram_to_pattern(spectrogram, target_size, device)

    # Return pattern + audio (for playback/saving)
    return pattern, audio_signal, sample_rate


def load_audio_pattern(audio_path, target_size=(256, 256), device="cpu"):
    """
    Load audio file and convert to spectrogram pattern.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        target_size: Desired pattern dimensions
        device: torch.device

    Returns:
        torch.Tensor: Spectrogram as 2D pattern
    """
    import soundfile as sf

    # Load audio file
    audio_signal, sample_rate = sf.read(audio_path)

    # If stereo, convert to mono
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal.mean(axis=1)

    # Convert to spectrogram
    spectrogram = audio_to_spectrogram(audio_signal, sample_rate)

    # Convert to pattern
    pattern = spectrogram_to_pattern(spectrogram, target_size, device)

    return pattern


def pattern_to_spectrogram(pattern, original_shape=None):
    """
    Convert substrate pattern back to spectrogram (for audio generation).

    STUB: Will be implemented for Experiment 1C (generative pathway).

    Args:
        pattern: 2D tensor from substrate
        original_shape: Original spectrogram dimensions (if different)

    Returns:
        numpy array: Spectrogram
    """
    # TODO: Implement in generative experiment
    # 1. Denormalize pattern
    # 2. Resize to original_shape if needed
    # 3. Return as numpy array for audio synthesis
    raise NotImplementedError("Spectrogram conversion not yet implemented")


def generate_random_pattern(width, height, density=0.1, device="cpu"):
    """
    Generate random pattern (for control experiments).

    Args:
        width: Pattern width
        height: Pattern height
        density: Fraction of pixels that are active
        device: torch.device

    Returns:
        torch.Tensor: Random pattern
    """
    pattern = torch.rand(height, width, device=device)
    pattern = (pattern < density).float() * 0.8
    return pattern


# Pattern registry: Easy access to all geometric patterns
GEOMETRIC_PATTERNS = [
    "square_spots",
    "ring",
    "line",
    "cross",
    "diagonal_stripes",
]


def create_pattern_set(pattern_types, width, height, device, offsets=None):
    """
    Create multiple patterns at once.

    Useful for multi-pattern experiments.

    Args:
        pattern_types: List of pattern type names
        width: Pattern width
        height: Pattern height
        device: torch.device
        offsets: Optional list of (x, y) offsets

    Returns:
        dict: Maps pattern type to pattern tensor
    """
    if offsets is None:
        offsets = [(0, 0)] * len(pattern_types)

    patterns = {}
    for ptype, offset in zip(pattern_types, offsets):
        patterns[ptype] = create_geometric_pattern(
            ptype, width, height, device, region_offset=offset
        )

    return patterns
