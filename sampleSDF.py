import numpy as np
import open3d as o3d
import pymeshlab
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

def compute_face_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)

    areas = 0.5 * np.linalg.norm(normals, axis=1)

    return areas

def compute_signed_distance_and_closest_goemetry(query_points, scene):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    direction = (query_points - closest_points['points'].numpy()) / distance[:, np.newaxis]

    rays = np.concatenate([query_points, direction], axis=-1)

    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points

def compute_rgb_at_uv(uv_coords, texture_image, bilinear = True):
    map_size = [texture_image.shape[0], texture_image.shape[1]]
    if not bilinear:
        # change from uv space to image space
        map_coords = np.column_stack((uv_coords[:, 0] * map_size[1], uv_coords[:, 1] * map_size[0])).astype(int)

        # flip the y image vertically and clamp
        clamped_coord_x = np.clip(map_coords[:, 0], 0, map_size[1] - 1)
        clamped_coord_y = map_size[0] - 1 - np.clip(map_coords[:, 1], 0, map_size[0] - 1)
        clamped_coord = np.column_stack((clamped_coord_x, clamped_coord_y))

        r = texture_image[clamped_coord[:, 1], clamped_coord[:, 0], 0]
        g = texture_image[clamped_coord[:, 1], clamped_coord[:, 0], 1]
        b = texture_image[clamped_coord[:, 1], clamped_coord[:, 0], 2]

        rgb = np.column_stack((r, g, b))
    else:
        # four nearest pixel
        x = uv_coords[:, 0] * (map_size[1] - 1)
        y = (1 - uv_coords[:, 1]) * (map_size[0] - 1)
        x0, y0, x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int), np.ceil(x).astype(int), np.ceil(y).astype(int)

        # clip to resolution range
        x0 = np.clip(x0, 0, map_size[1] - 1)
        x1 = np.clip(x1, 0, map_size[1] - 1)
        y0 = np.clip(y0, 0, map_size[0] - 1)
        y1 = np.clip(y1, 0, map_size[0] - 1)

        weight_x1 = x - x0
        weight_x0 = 1 - weight_x1
        weight_y1 = y - y0
        weight_y0 = 1 - weight_y1
        weight_x0 = weight_x0.reshape(-1, 1)
        weight_x1 = weight_x1.reshape(-1, 1)
        weight_y0 = weight_y0.reshape(-1, 1)
        weight_y1 = weight_y1.reshape(-1, 1)


        pixel00 = texture_image[y0, x0]
        pixel01 = texture_image[y1, x0]
        pixel10 = texture_image[y0, x1]
        pixel11 = texture_image[y1, x1]

        rgb =  (pixel00 * weight_x0 * weight_y0 +
                pixel01 * weight_x0 * weight_y1 +
                pixel10 * weight_x1 * weight_y0 +
                pixel11 * weight_x1 * weight_y1)
    return rgb


def SampleFromSurface(vertices, faces, face_area, face_selected, num_sampled_points, var1, var2):
    # Generate random values u and v
    u = np.random.rand(num_sampled_points, 1)
    v = np.random.rand(num_sampled_points, 1)

    # Calculate barycentric coordinates w0, w1, and w2
    w0 = 1 - np.sqrt(u)
    w1 = np.sqrt(u) * (1 - v)
    w2 = np.sqrt(u) * v

    # Calculate probabilities for face sampling
    probabilities = face_area[face_selected] / np.sum(faceArea[face_selected])

    # Sample face indices using the probability distribution
    sample_face_idxs = np.random.choice(face_selected, num_sampled_points, p=probabilities)

    # Calculate sample points
    v0 = vertices[faces[sample_face_idxs, 0], :]
    v1 = vertices[faces[sample_face_idxs, 1], :]
    v2 = vertices[faces[sample_face_idxs, 2], :]
    surface_points = v0 * w0 + v1 * w1 + v2 * w2

    noise1 = np.random.normal(0, var1**0.5, (num_sampled_points, 3))
    noise2 = np.random.normal(0, var2**0.5, (num_sampled_points, 3))
    sampled_points = np.vstack((surface_points + noise1, surface_points + noise2))

    return sampled_points


def SampleFromBoundingCube(num_rand_points, cube_length = 1):
    sampled_points = np.random.uniform(-cube_length / 2, cube_length / 2, (num_rand_points, 3))
    return sampled_points


totalSampleNumber = 500000
# not the var in paper "deepsdf" these values are set for better visualization
noiseVar1 = 0.00005
noiseVar2 = 0.000005
num_samp_near_surf_ratio = 47 / 50

visualizationWithColor = True
visualizationWithDistance = True

ref_obj_path = './deadRose/deadRose.obj'
ref_tex_path = './deadRose/deadRose.jpg'
ms = pymeshlab.MeshSet()
ms.load_new_mesh(ref_obj_path)

refV = ms.current_mesh().vertex_matrix()
refF = ms.current_mesh().face_matrix()
refVN = ms.current_mesh().vertex_normal_matrix()
refUV = ms.current_mesh().wedge_tex_coord_matrix().reshape(refF.shape[0], 3, 2)
refFN = ms.current_mesh().face_normal_matrix()

print('load mesh vertices: {}'.format(refV.shape))
print('load mesh faces: {}'.format(refF.shape))
print('load mesh vertex normal: {}'.format(refVN.shape))
print('load mesh face normal: {}'.format(refFN.shape))
print('load mesh uv coordinates: {}'.format(refUV.shape))

# normalize to unit sphere
box = ms.current_mesh().bounding_box()
box_center = 0.5 * (box.max() + box.min())
refV = (refV - box_center) / box.diagonal()

faceArea = compute_face_areas(refV, refF)

refTexture = np.array(Image.open(ref_tex_path))
refMesh = o3d.geometry.TriangleMesh()
refMesh.vertices = o3d.utility.Vector3dVector(refV)
refMesh.triangles = o3d.utility.Vector3iVector(refF)
refMesh.triangle_normals = o3d.utility.Vector3dVector(refFN)

num_samp_near_surf = int(num_samp_near_surf_ratio * totalSampleNumber)
# you can also sample from partial faces
fSelected = [i for i in range(0, refF.shape[0])]
surfSamples = SampleFromSurface(refV, refF, faceArea, fSelected, int(num_samp_near_surf / 2), noiseVar1, noiseVar2)
randSamples = SampleFromBoundingCube(totalSampleNumber - num_samp_near_surf, 1)
totalSamples = np.vstack((surfSamples, randSamples))
totalSamples = totalSamples.astype(np.float32)

# get SDF values
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(refMesh))
refD, refIntersectionInfo = compute_signed_distance_and_closest_goemetry(totalSamples, scene)

# you can use refIntersectionInfo to calculate nearest points, colors ...
refIntersectionTris = refF[refIntersectionInfo['primitive_ids'].numpy()]
# (SampleNumber, 3, 2)
refIntersectionUVs = refIntersectionInfo['primitive_uvs'].numpy()
refIntersectionTriUVs = refUV[refIntersectionInfo['primitive_ids'].numpy(), :, :]
refIntersectionTextureUVs = refIntersectionTriUVs[:, 0, :].reshape(totalSampleNumber, 2) * (
            1 - refIntersectionUVs[:, 0].reshape(-1, 1) - refIntersectionUVs[:, 1].reshape(-1, 1)) + \
                            refIntersectionTriUVs[:, 1, :].reshape(totalSampleNumber, 2) * refIntersectionUVs[:,
                                                                                      0].reshape(-1, 1) + \
                            refIntersectionTriUVs[:, 2, :].reshape(totalSampleNumber, 2) * refIntersectionUVs[:,
                                                                                      1].reshape(-1, 1)
# v0 + (v1 - v0) * u + (v2 - v0) * v
refIntersectionPoints = refIntersectionInfo['points'].numpy()
refIntersectionColors = compute_rgb_at_uv(refIntersectionTextureUVs, refTexture, True)

# samplePointXYZ NerghborhoodPointXYZ RGB SDF
sdfSamplesWithColor = np.hstack((totalSamples, refIntersectionPoints, refIntersectionColors, refD.reshape(-1, 1)))
if visualizationWithColor:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(totalSamples)
    point_cloud.colors = o3d.utility.Vector3dVector(refIntersectionColors / 255)
    o3d.visualization.draw_geometries([point_cloud])
if visualizationWithDistance:
    colors = [(1, 0, 0), (0, 1, 0)]
    cmap = LinearSegmentedColormap.from_list('CustomCmap', colors, N=256)
    tmpD = abs(refD) / max(refD)
    tmpColor = cmap(tmpD)[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(totalSamples)
    point_cloud.colors = o3d.utility.Vector3dVector(tmpColor)
    o3d.visualization.draw_geometries([point_cloud])

sdfSamples = np.hstack((totalSamples, refD.reshape(-1, 1)))
sdf_pos = sdfSamples[sdfSamples[:, 3] >= 0]
sdf_neg = sdfSamples[sdfSamples[:, 3] < 0]

sdfFileName = "./data/deadrose.npz"
np.savez(sdfFileName, pos=sdf_pos, neg=sdf_neg)





