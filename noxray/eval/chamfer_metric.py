import torch
import numpy as np

from tk3dv.extern.chamfer import ChamferDistance
from tk3dv.nocstools import datastructures as ds 

class ChamferMetric(object):

    def __init__(self, device='cuda', torch_device=None):
        if torch_device is None:
            self.device = torch.device(device) # 'cuda' or 'cpu'
        else:
            self.device = torch_device
        self.ChamferDist = ChamferDistance()

    def chamfer_dist(self, pc1, pc2):
        ''' Finds the mean chamfer distance between two (Nx3) numpy point clouds. '''
        pc1 = torch.from_numpy(pc1.astype(np.float32)).to(self.device)
        pc2 = torch.from_numpy(pc2.astype(np.float32)).to(self.device)

        # outputs minimum squared distance for each point in the point cloud
        # expects inputs of shape [B, N, 3]
        dist1, dist2 = self.ChamferDist(pc1.unsqueeze(0), pc2.unsqueeze(0))
        mean_dist = torch.mean(dist1) + torch.mean(dist2)
        return mean_dist.item()

    def chamfer_dist_nocs(self, nocs_list0, nocs_list1):
        ''' Takes in two tuples of numpy NOCS maps, and finds the chamfer distance between their unions. '''
        # create union point clouds
        nocs_pc0 = self.nocs2pc(nocs_list0)
        nocs_pc1 = self.nocs2pc(nocs_list1)
        return self.chamfer_dist(nocs_pc0, nocs_pc1)

    def nocs2pc(self, nocs_list):
        ''' Turns a tuple of NOCS maps into a combined point cloud '''
        nocs_pc = []
        for nocs_map in nocs_list:
            nocs = ds.NOCSMap3D(nocs_map)
            nocs_pc.append(nocs.Points)
        nocs_pc = np.concatenate(nocs_pc, axis=0)
        return nocs_pc
    
        