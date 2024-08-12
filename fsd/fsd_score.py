"""
TODO DOCUMENTATION
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
from residual2plus1 import R2Plus1D


def calculate_activation_statistics(vids, model, batch_size=32, dims=512, cuda=True, normalize=True, verbose=0):
    model.eval()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    with torch.no_grad():
        features = []
        dataloader = DataLoader(
                vids,
                batch_size=batch_size,
                shuffle=False
        )
        if verbose > 0:
            iter_dataset = tqdm(dataloader, dynamic_ncols=True)
        else:
            iter_dataset = dataloader

        for videos in iter_dataset:
            videos = videos.type(torch.FloatTensor).to(device)
            activation = model(videos)
            if activation.shape[2] != 1 or activation.shape[3] != 1:
                activation = F.adaptive_avg_pool2d(activation, output_size=(1, 1))
            features.append(activation.cpu().numpy().reshape(-1, dims))

        features = np.concatenate(features, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
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

    return (diff.dot(diff) +
            np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fsd_score(r_imgs, g_imgs, batch_size=50, dims=512, cuda=True, normalize=True):
    model = R2Plus1D()

    # todo r_cache

    m1, s1 = calculate_activation_statistics(r_imgs, model, batch_size, dims, cuda, normalize)
    m2, s2 = calculate_activation_statistics(g_imgs, model, batch_size, dims, cuda, normalize)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


if __name__ == "__main__":
    from fsd.dataset import CustomDataset

    test1_dataset = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test1")
    test2_dataset = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2")

    fsd_value = fsd_score(test1_dataset, test2_dataset)
    print('FSD:', fsd_value)
