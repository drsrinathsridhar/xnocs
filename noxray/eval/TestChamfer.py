import torch
import numpy as np
from tk3dv.extern.chamfer import ChamferDistance
chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are n_points x 3 matrices

points = torch.from_numpy(np.array([[[1, 1, 0], [2, 1, 0]]], dtype=np.float32))
#points.to(torch.device('cuda'))
points_reconstructed = torch.from_numpy(np.array([[[2, 2, 0]]], dtype=np.float32))
#points_reconstructed.to(torch.device('cuda'))

dist1, dist2 = chamfer_dist(points, points_reconstructed)
# outputs minimum squared distance for each point in the point cloud
print(dist1)
print(dist2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))
print('loss: ' + str(loss))
total_dist = dist1 + dist2
cham = torch.sum(dist1) + torch.sum(dist2)
print('chamfer dist: ' + str(cham))