# DropBlock

> NOTICE : This implementation is for my study and practice. It may not be perfect on your purpose.

I've implement dropblock based on pytorch. DropBlock is advanced version of conventional dropout methods. It subducts not only hidden nodes selected but others in the blocks with the center being the selected node while dropout remove only hidden nodes selected randomly.

In the paper, the authors had mentioned DropBlock with constant the probability hurts training. This problem can be alleviated by keeping the probability change with linear scheme.

This dropblock implementation supports data with 1d and 2d.

- Use dropblock with constant the probability of removal
    - Import `dB.py` only.
- Use dropBlock with the scheduled probability
    - `from dBScheduler import probScheduler`
    - `from dB import dropBlock`

## Reference

I've refer some materials mentioned in below...

- [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)
- [https://github.com/miguelvr/dropblock](https://github.com/miguelvr/dropblock)
