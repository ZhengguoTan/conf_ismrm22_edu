# %%
import sigpy as sp
from sigpy import linop
import random

import os.path
import sys
sys.path.insert(0, os.path.join(os.environ['BART_PATH'], 'python'))
import cfl

# %%
I = sp.shepp_logan((256, 256))

block_shape = (16, 16)
stride_shape = (16, 16)

axes = range(-2, 0)
shift = [random.randint(0, block_shape[s]) for s in axes]
print('>>> shift: ' + str(shift))

RandShift = linop.Circshift(I.shape, shift, axes)

A = linop.ArrayToBlocks(I.shape, block_shape, stride_shape)

shift = RandShift * I
cfl.writecfl('shift', shift)

blocks = A * shift
cfl.writecfl('blocks', blocks)
