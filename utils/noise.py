import numpy as np
import cv2
import torch

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def corrcoef(tensor, rowvar=True, clip=False):
    """
    Get Pearson product-moment correlation coefficients (np.corrcoef).
    
    Args:
        tensor (Tensor): Image tensor of shape (B, H*W, K*K, C).
    
    Returns:
        covariance (Tensor): Tensor of shape (B, H*W, K*K, K*K).
    """

    x = tensor.to(torch.float32)
    covariance = cov(x, rowvar=rowvar)
    variance = covariance.diagonal(0, -1, -2)
    if variance.is_complex():
        variance = variance.real
    stddev = variance.sqrt() + 1e-8
    covariance = covariance / stddev.unsqueeze(-1)
    covariance = covariance / stddev.unsqueeze(-2)
    if clip:
        if covariance.is_complex():
            covariance.real.clip_(-1, 1)
            covariance.imag.clip_(-1, 1)
        else:
            covariance.clip_(-1, 1)
    return covariance.to(tensor.dtype)

def corrcoef_optimized(x, rowvar=True, clip=False):
    """
    Get Pearson product-moment correlation coefficients (np.corrcoef).
    
    Args:
        tensor (Tensor): Image tensor of shape (B, H*W, K*K, C).
    
    Returns:
        covariance (Tensor): Tensor of shape (B, H*W, K*K, K*K).
    """
        
    # Ensure working in float32 for stability.
    # x = tensor.to(torch.float32)
    if not rowvar:
        x = x.transpose(-1, -2)
    
    # Subtract mean and compute standard deviation along the last dimension.
    mean = x.mean(dim=-1, keepdim=True)
    x = x - mean
    std = x.std(dim=-1, unbiased=True, keepdim=True) + 1e-4
    
    # Normalize the tensor.
    x_norm = x / std
    
    # Compute the correlation matrix as a dot product of the normalized tensor.
    corr = x_norm @ x_norm.transpose(-1, -2)
    
    # Optionally clip the values.
    if clip:
        corr = corr.clamp(-1, 1)
    
    return corr # .to(tensor.dtype)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    
    return images


def extract_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    patches = patches.reshape(batch_size, channel, -1, patches.shape[-1])

    return patches  # [N, C, k*k, L], L is the total number of such blocks

# https://github.com/haoyuc/MaskedDenoising/blob/9cd4c62a7a82178d86f197e11f2d0ba3ab1fbd5a/utils/utils_mask.py#L379

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)


def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2*sf
    if random.random() < 0.5:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,11)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5/sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2*random.random()+2.0)  # [2, 4]
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    return img

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):
    return np.float32(img/255.)

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img
    
def add_correlated_Gaussian_noise(img, noise_level1=2, noise_level2=25, filter_size=3, generator=None):
    if generator is None:
        rng = np.random.default_rng()
    else:
        rng = generator

    if noise_level1 == noise_level2:
        noise_level = noise_level1
    else:
        noise_level = rng.integers(noise_level1, noise_level2, size=1)

    n = rng.normal(0.0, noise_level / 255.0, img.shape).astype(np.float32)
    n = ndimage.uniform_filter(n, size=filter_size)
    result = np.clip(img + n, 0.0, 1.0)

    return result

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25, generator=None, channel_wise=False):
    if generator is None:
        rng = np.random.default_rng()
    else:
        rng = generator

    C, H, W = img.shape
    if channel_wise:
        if noise_level1 == noise_level2:
            noise_level = noise_level1
        else:
            noise_level = rng.integers(noise_level1, noise_level2, size=3)
        n = np.concatenate([rng.normal(0.0, noise_level[i] / 255.0, (1, H, W)).astype(np.float32) for i, _ in enumerate(range(C))], axis=0)
    else:
        if noise_level1 == noise_level2:
            noise_level = noise_level1
        else:
            noise_level = rng.integers(noise_level1, noise_level2, size=1)
    
        n = rng.normal(0.0, noise_level / 255.0, (C, H, W)).astype(np.float32)
    result = np.clip(img + n, 0.0, 1.0)

    return result, noise_level

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean