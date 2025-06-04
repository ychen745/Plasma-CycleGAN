import numpy as np
import torch
from pytorch_msssim import ssim

# Note that we use the axial slice-wise performance metrics to evaluate all competing methods
# and ours because the axial axis is mainly used in real clinical scenarios.

def mean_absolute_error(image_true, image_generated):
    """Compute mean absolute error.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        mse: (float) mean squared error
    """
    image_true = image_true.squeeze(0).squeeze(0)
    image_generated = image_generated.squeeze(0).squeeze(0)
    losses = 0.
    image_size = min(image_true.shape)
    for i in range(image_size):
        losses += torch.abs(image_true[i, :, :] - image_generated[i, :, :]).mean()
    losses /= image_size
    return losses


def peak_signal_to_noise_ratio(image_true, image_generated):
    """"Compute peak signal-to-noise ratio.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        psnr: (float) peak signal-to-noise ratio"""
    image_true = image_true.squeeze(0).squeeze(0)
    image_generated = image_generated.squeeze(0).squeeze(0)
    losses = 0.
    image_size = min(image_true.shape)
    for i in range(image_size):
        losses += -10 * np.log10(((image_true[i, :, :] - image_generated[i, :, :]) ** 2).mean().cpu())
    losses /= image_size
    return losses


def structural_similarity_index(image_true, image_generated):
    """Compute structural similarity index.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        ssim: (float) structural similarity index """
    image_true = image_true.squeeze(0).squeeze(0)
    image_generated = image_generated.squeeze(0).squeeze(0)
    losses = 0.
    image_size = min(image_true.shape)
    for i in range(image_size):
        losses += ssim(image_generated[i, :, :].unsqueeze(0).unsqueeze(0),
                       image_true[i, :, :].unsqueeze(0).unsqueeze(0), data_range=1, size_average=True)
    losses /= image_size
    return losses
