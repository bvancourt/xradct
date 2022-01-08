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
      
    
class XRadCT(CTScan):
    def __init__(self, voxels, name):
        super().__init__(voxels, [.1, .1, .1], name)
        self.x, self.y, self.z = voxelhelp.index_grid(voxels)

    @classmethod
    def from_dicom_files(cls, path_to_files):
        voxels = voxelhelp.voxels_from_dicom_folder(path_to_files)
        name = path_to_files.split('\\')[-1].split('/')[-1]
        return cls(voxels, name)
    
    def make_bed_air(self):
        pass
    
class OneMouseScan(XRadCT):
    pass

class OneRowScan(XRadCT):
    def __init__(self, voxels, name):
        super().__init__(voxels, name)
        self.mouseIDs = [with_spaces.replace(' ', '') for with_spaces in name.split(',')]

    def to_individual_mice(self):
        def cosine_window(length):
            t = np.arange(1, length)
            return (1-np.cos(2*np.pi*t/length))/length

if __name__ == '__main__':
    print('You\'re really supposed to import this file, not run it.')
