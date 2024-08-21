"""
TODO DOCUMENTATION
Calculates the Fréchet Story Distance (FSD) to evaluate consistency in the sequence of images

Code adapted from https://github.com/bioinf-jku/TTUR
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
from residual2plus1 import R2Plus1D


def _compute_activation(vids, model, batch_size=32, dims=512, cuda=True):
    """Calculates the activations of the pool_3 layer of the model for all frames"""

    model.eval()
    if cuda: device = torch.device('cuda')
    else: device = torch.device('cpu')
    model.to(device)

    with torch.no_grad():
        features = []
        dataloader = DataLoader(
                vids,
                batch_size=batch_size,
                shuffle=False
        )
        for videos in dataloader:
            videos = videos.type(torch.FloatTensor).to(device)
            activation = model(videos)
            if activation.shape[2] != 1 or activation.shape[3] != 1:
                activation = F.adaptive_avg_pool2d(activation, output_size=(1, 1))
            features.append(activation.cpu().numpy().reshape(-1, dims))
        features = np.concatenate(features, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

    return mu, sigma


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    TODO
    Numpy implementation of the Fréchet Distance.
    The Fréchet distance between two multivariate Gaussian distributions:
        X_1 ~ N(μ_1, C_1)
    and
        X_2 ~ N(μ_2, C_2)
    is
        d^2 = ||μ_1 - μ_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Also known as 2-Wasserstein distance on real numbers

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : activations of the pool_3 layer of the evaluation model for generated samples.
    -- mu2   : activations of the pool_3 layer of the evaluation model for representative data set.
    -- sigma1: covariance over activations of the pool_3 layer for generated samples.
    -- sigma2: covariance over activations of the pool_3 layer for representative data set.

    Returns:
    --   : The Fréchet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)  # geometric mean of activation covariances
    if not np.isfinite(covmean).all():
        print('fid calculation produces singular product; '
              'adding %s to diagonal of cov estimates') % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def fsd_score(ground_truth, generated, batch_size=32, dims=512, cuda=True):
    model = R2Plus1D()

    m1, s1 = _compute_activation(ground_truth, model, batch_size, dims, cuda)
    m2, s2 = _compute_activation(generated, model, batch_size, dims, cuda)
    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


if __name__ == "__main__":
    from fsd.dataset import CustomDataset

    set1_dataset1 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test1\\set_1")
    set2_dataset1 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test1\\set_2")

    fsd_value1 = fsd_score(set1_dataset1, set2_dataset1)
    print('Test 1 FSD:', fsd_value1)

    set1_dataset2 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_1")
    set2_dataset2 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_2")

    fsd_value2 = fsd_score(set1_dataset2, set2_dataset2)
    print('Test 2 FSD:', fsd_value2)
