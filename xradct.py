import numpy as np
import importlib
import warnings
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import linalg

import voxelhelp
importlib.reload(voxelhelp)

class CTScan(): # should be compatible with both XRad and Quantum GXII uCT
    def __init__(self, voxels, voxel_spacing, name):
        self.name = name
        self.spacing = voxel_spacing # in millimeters
        self.voxels = voxels

if __name__ == '__main__':
    print('You\'re really supposed to import this file, not run it.')
