from fsd_score import fsd_score
from dataset import CustomDataset
import functools


if __name__ == '__main__':
    # common params
    RES = (64, 64)
    custom_dataset = functools.partial(CustomDataset, res=RES)

    gen = custom_dataset("<path>")
    gt = custom_dataset("<path>")
    fsd = fsd_score((gen, gt))
    print('FSD score:', fsd)
