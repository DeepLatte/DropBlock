import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class dropBlock(nn.Module):
    def __init__(self, prob, block_size):
        super(dropBlock, self).__init__()
        '''
        Args
            - prob (float) : drop rate(contrast with keep_prob mentioned in the orginal paper.)
            - block_size (int) : Size of the square mask.
        '''
        if prob > 1 or prob < 0:
            raise ValueError('drop rate (prob) shoud be in 0 <= p <= 1')
        self.prob = prob
        self.block_size = block_size

    def forward(self, input):
        '''
        input : (B, C, T)
        self.Training can check whether in training mode or eval mode.
        '''
        if len(input.size()) == 3:
            if (not self.training) or (self.prob == 0):
                output = input
            else:
                gamma = self.calGamma(self.prob, self.block_size, input.size(-1), 1)
                mask = (torch.rand(1, *input.size()[1:]) < gamma).float().to(input.device)
                mask = F.max_pool1d(input=mask, kernel_size=self.block_size,
                                    padding=self.block_size//2, stride=1)
                mask = 1 - mask
                output = torch.einsum('ijk, jk -> ijk', input, mask[0])
                if mask.sum() != 0:
                    output = output * mask.numel() / mask.sum()

        elif len(input.size()) == 4:
            if (not self.training) or (self.prob == 0):
                output = input
            else:
                gamma = self.calGamma(self.prob, self.block_size, input.size(-1), 2)
                mask = (torch.rand(1, *input.size()[1:]) < gamma).float().to(input.device)
                mask = F.max_pool2d(input=mask, kernel_size=(self.block_size, self.block_size),
                                    padding=self.block_size//2, stride=1)
                mask = 1 - mask
                output = torch.einsum('ijkl, jkl -> ijkl', input, mask[0])
                if mask.sum() != 0:
                    output = output * mask.numel() / mask.sum()

        return output

    @staticmethod
    def calGamma(prob, block_size, feat_size, n):
        return prob*(feat_size**n)/((block_size**n)*(feat_size-block_size+1)**n)

# if __name__ == "__main__":
#     prob = 0.3
#     block_size = 5
#     x = torch.ones([5, 10, 125, 125])
#     drop = dropBlock(prob, block_size)
#     drop.train()
#     for _ in range(100):
#         drop(x)