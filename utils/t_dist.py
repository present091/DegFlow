import scipy.stats as stats
import numpy as np
import torch
import math

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)
    
def sample_t(exponential_pdf, num_samples, a):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t


def sample_t_uniform(num_samples, t_min=1e-5, t_max=1-1e-5, device=None):
    """
    Args:
        num_samples (int): 생성할 t의 개수
        t_min (float): 최소값 (포함)
        t_max (float): 최대값 (포함)
        device (torch.device or str, optional): 생성할 텐서의 디바이스

    Returns:
        torch.FloatTensor: shape = (num_samples,), uniform([t_min, t_max]) 샘플
    """
    if device is None:
        device = torch.device('cpu')
    # 0~1 균등분포에서 샘플링 후 스케일링
    t = torch.rand(num_samples, device=device) * (t_max - t_min) + t_min
    return t

if __name__ == '__main__':
    # Create an instance of the class
    exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')

    num_samples = 100000
    a = 2
    t_positions = torch.tensor([0.0, 1.0/3.0, 1.0])
    samples = sample_t_piecewise_exponential(exponential_distribution, t_positions, num_samples, a).numpy()
    # samples = sample_t(exponential_distribution, num_samples, a).numpy()
    # samples = sample_t_beta(num_samples)

    # Plot the histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=1000, density=True)
    plt.savefig('exponential_samples.png', dpi=300)