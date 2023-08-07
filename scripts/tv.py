# %%
import sigpy as sp
from sigpy import linop

import os.path
import sys
sys.path.insert(0, os.path.join(os.environ['BART_PATH'], 'python'))
import cfl

# %% Total Variation (TV) along x and y directions
I = sp.shepp_logan((256, 256))

G = linop.FiniteDifference(I.shape, axes=[-2, -1])

y = G * I

cfl.writecfl('shepp_logan', I)
cfl.writecfl('shepp_logan_TV', y)


