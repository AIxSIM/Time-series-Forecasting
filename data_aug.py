import numpy as np
import torch

def time_warping(data, sigma=0.2):
    """Apply time warping to a time-series."""
    num_points = data.shape[0]
    time_steps = np.arange(num_points)
    warp_factors = np.random.normal(1, sigma, num_points)
    return np.interp(time_steps, time_steps * warp_factors, data)

def jittering(data, sigma=0.1):
    """Add random noise to a time-series."""
    # noise = np.random.normal(0, sigma, np.array(data).shape)
    # return data + noise

    """
    Apply jittering to the sequence tensor in the data.
    Args:
        data: A list containing two tensors: [sequence, target]
        sigma: Standard deviation for the noise.
    """
    sequence, target = data  # Unpack the list

    # Apply jittering only to the sequence tensor
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence, dtype=torch.float32)

    # Generate noise and add to the sequence
    noise = torch.normal(0, sigma, size=sequence.size())
    augmented_sequence = sequence + noise

    # Return the augmented sequence and the original target
    return [augmented_sequence, target]

def augment_data(data):
    """Create positive pairs by applying augmentation techniques."""
    augmented_data = jittering(data)  # Apply jittering as an example

    return augmented_data