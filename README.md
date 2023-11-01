# DeepSDF python implements of sampling
The original environment in [Deepsdf](https://github.com/daydreamer2023/DeepSDF) to pre-process the data is hard to configure. So this project rewrite the sampling process to sample sdf values from mesh objects using [pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab) and [open3d](http://www.open3d.org/).

# How to Use
Install the environments using Anaconda or pip and run `python sampleSdf.py`, you can change `ref_obj_path`, `ref_tex_path` to your own mesh path and change sampling numbers `totalSampleNumber` and the noise ratio `noiseVar1,2`, and you can visualize the samples with the nearest point color `visualizationWithColor` or color controlled by distance `visualizationWithDistance`.

# Example
The original mesh is at `data/deadRose/deadRose.obj` 
![image](https://github.com/yyyykf/DeepSDF-Functions/blob/main/data/originalMesh.png) 
the result is saved to `data/deadrose.npz`
the visualization of the sampled sdfs:
![image](https://github.com/yyyykf/DeepSDF-Functions/blob/main/data/sampledPointsWithColor.png) 
![image](https://github.com/yyyykf/DeepSDF-Functions/blob/main/data/sampledPointsWithDistance.png) 
