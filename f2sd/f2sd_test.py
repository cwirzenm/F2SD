from f2sd_score import f2sd_score
from residual2plus1 import R2Plus1D
import functools


if __name__ == '__main__':
    SEQ_LENGTH = 3

    MODEL = R2Plus1D()
    RESOLUTION = (256, 256)
    DIMS = 512
    MODE = 'R2+1D'

    f2sd_func = functools.partial(f2sd_score, seq_length=SEQ_LENGTH, model=MODEL, dims=DIMS, res=RESOLUTION, mode=MODE)

    data = "<path>"
    f2sd = f2sd_func(data)
    print('F2SD score:', f2sd, end='\n\n')
