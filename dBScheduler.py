import torch
import torch.nn as nn
import numpy as np

from dropBlock import dropBlock

class probScheduler(nn.Module):
    def __init__(self, dropBlock, start_prob, end_prob, iteration):
        super(probScheduler, self).__init__()
        self.dropBlock = dropBlock
        self.i = 0
        self.prob_schedule = np.linspace(start_prob, end_prob, iteration)
    
    def forward(self, input):
        return self.dropBlock(input)

    def step(self):
        if self.i < len(self.prob_schedule):
            self.dropBlock.prob = self.prob_schedule[self.i]
        
        self.i += 1

if __name__ == "__main__":
    start_prob = 0
    end_prob = 0.25
    block_size = 5
    x = torch.ones([5, 10, 125, 125])
    drop = dropBlock(start_prob, block_size)
    pS = probScheduler(drop, start_prob, end_prob, 100)
    drop.train()
    for _ in range(100):
        pS.step()
        pS(x)
        