"""Computational Neuroscience Decision-Making Package.

Modules:
    data_loader: Download and load behavioural data
    preprocessing: Clean and filter RT data
    features: Extract features from RT distributions
    ddm_model: Drift diffusion model implementation
    decoding: Machine learning classifiers
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

from . import data_loader
from . import preprocessing
from . import features
from . import ddm_model
from . import decoding
