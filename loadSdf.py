import open3d as o3d
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


sdfFileName = "./data/deadrose.npz"
sdfValues = np.load(sdfFileName)
sdf_pos = sdfValues["pos"]
sdf_neg = sdfValues["neg"]

colors = [(1, 0, 0), (0, 1, 0)]
cmap = LinearSegmentedColormap.from_list('CustomCmap', colors, N=256)
for sdf in [sdf_pos, sdf_neg]:
    xyz = sdf[:, :3]
    refD = sdf[:, 3]
    tmpD = abs(refD) / max(refD)
    tmpColor = cmap(tmpD)[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(tmpColor)
    o3d.visualization.draw_geometries([point_cloud])