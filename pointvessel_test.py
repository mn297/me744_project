# %%
import numpy as np

import open3d as o3d
# %%
DATA_DIR = 'pointvessel_data'
pcl = np.load(f'{DATA_DIR}/dvn/0001/pcl_2048/pcl_0001.npy')

print(pcl.shape, pcl.dtype)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcl)
o3d.visualization.draw_geometries([pcd])

# %%