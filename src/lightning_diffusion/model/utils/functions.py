import torch
import numpy as np

def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Approximate the CDF of the standard normal distribution.

    Refert to https://github.com/PixArt-alpha/PixArt-alpha/blob/master/
    diffusion/model/diffusion_utils.py
    """
    return 0.5 * (1.0 + torch.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(
    x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """Compute the log-likelihood of a Gaussian distribution discretizing.

    Refer to: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/
    diffusion/model/gaussian_diffusion.py
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,  # noqa: PLR2004
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,  # noqa: PLR2004
                    torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs