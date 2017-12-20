import sys
sys.path.append("..")
import theano.tensor as TT
import numpy as np
from distributions.base import Distribution
from distributions.diagonal_gaussian import DiagonalGaussian

RecurrentDiagonalGaussian = DiagonalGaussian
