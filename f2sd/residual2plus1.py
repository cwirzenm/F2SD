import torch
import torch.nn as nn
from torchvision.models.video.resnet import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1D(nn.Module):

    def __init__(self):
        super(R2Plus1D, self).__init__()

        video_resnet = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1, progress=True)
        block: list = [
                video_resnet.stem,
                video_resnet.layer1,
                video_resnet.layer2,
                video_resnet.layer3,
                video_resnet.layer4,
                video_resnet.avgpool,
        ]
        self.blocks = nn.Sequential(*block)

    def forward(self, inputs) -> torch.Tensor:
        return self.blocks(inputs)
