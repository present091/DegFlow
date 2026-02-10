import torch
import lpips
import numpy as np
import torch.nn.functional as F
import cv2

from .image import rgb2ycbcr_pt
from .common import frozen_module


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/metrics/psnr_ssim.py#L52
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))

def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate SSIM (structural similarity) (PyTorch version).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim

def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])

class LPIPS:
    
    def __init__(self, net: str) -> None:
        self.model = lpips.LPIPS(net=net)
        frozen_module(self.model)
    
    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, normalize: bool) -> torch.Tensor:
        """
        Compute LPIPS.
        
        Args:
            img1 (torch.Tensor): The first image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            img2 (torch.Tensor): The second image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            normalize (bool): If specified, the input images will be normalized from [0, 1] to [-1, 1].
            
        Returns:
            lpips_values (torch.Tensor): The lpips scores of this batch.
        """
        return self.model(img1, img2, normalize=normalize)
    
    def to(self, device: str) -> "LPIPS":
        self.model.to(device)
        return self


# https://github.com/zsyOAOA/DANet

def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: 1 x C x H x W, torch array
        sigma_real: 1 x C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1.)
    return distance

def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=3).to(img_gt.device)
    sigma = gaussblur(err2, kernel, win_size, chn=3)
    sigma.clamp_(min=1e-10)

    return sigma

def calculate_alkd(real_noisy, fake_noisy, clean):
    sigma_real = estimate_sigma_gauss(real_noisy, clean)
    sigma_fake = estimate_sigma_gauss(fake_noisy, clean)
    alkd = kl_gauss_zero_center(sigma_fake, sigma_real)
    return alkd.item()

# https://github.com/BorealisAI/noise_flow/

# def calculate_kld(p_data, q_data, bin_edges=None, left_edge=-1.0, right_edge=1.0, n_bins=256):
#     """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
#     p_data = p_data.cpu()
#     q_data = q_data.cpu()

#     if bin_edges is None:
#         data_range = right_edge - left_edge
#         bin_width = data_range / n_bins
#         bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
#     p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
#     q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
#     idx = (p > 0) & (q > 0)
#     p = p[idx]
#     q = q[idx]
#     logp = np.log(p)
#     logq = np.log(q)
#     kl_fwd = np.sum(p * (logp - logq))
#     kl_inv = np.sum(q * (logq - logp))
#     kl_sym = (kl_fwd + kl_inv) / 2.0
#     return kl_fwd # , kl_inv, kl_sym

# def get_histogram(data, bin_edges=None, left_edge=-1.0, right_edge=1.0, n_bins=256):
#     data_range = right_edge - left_edge
#     bin_width = data_range / n_bins
#     if bin_edges is None:
#         bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
#     bin_centers = bin_edges[:-1] + (bin_width / 2.0)
#     n = np.prod(data.shape)
#     hist, _ = np.histogram(data, bin_edges)
#     return hist / n, bin_centers



def get_histogram(data, bin_edges=None, left_edge=-1.0, right_edge=1.0, n_bins=256):
    """
    Compute a normalized histogram of `data` over [left_edge, right_edge] with n_bins.
    Returns:
      hist:  shape (n_bins,), sums to 1
      bin_centers: shape (n_bins,)
    """
    # ensure numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy().ravel()
    else:
        data = np.asarray(data).ravel()

    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.linspace(left_edge, right_edge, n_bins + 1)
    hist, edges = np.histogram(data, bins=bin_edges, density=False)
    hist = hist.astype(np.float64)
    hist /= hist.sum()  # normalize to sum = 1
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    return hist, bin_centers

def calculate_kld(p_data, q_data,
                  bin_edges=None,
                  left_edge=-1.0, right_edge=1.0,
                  n_bins=256,
                  eps=1e-8):
    """
    Compute forward, inverse, and symmetric KL divergences between two sample sets p_data and q_data.
    Applies epsilon‐smoothing and full‐support normalization to avoid zero‐division or infinite terms.
    Returns:
      kl_fwd:    KL(p || q)
      kl_inv:    KL(q || p)
      kl_sym:    0.5 * (kl_fwd + kl_inv)
    """
    # build histograms
    p_hist, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q_hist, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)

    # epsilon smoothing and renormalize
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    # compute divergences
    kl_fwd = np.sum(p_hist * np.log(p_hist / q_hist))
    # kl_inv = np.sum(q_hist * np.log(q_hist / p_hist))
    # kl_sym = 0.5 * (kl_fwd + kl_inv)

    return kl_fwd