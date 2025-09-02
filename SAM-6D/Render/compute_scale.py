import argparse
import json
import os

import numpy as np
import trimesh

from scipy.spatial import ConvexHull, distance_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help="The path to dataset")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
args = parser.parse_args()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
    print(radius)

    return 1/(2*radius)

def maximum_width(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    distances = distance_matrix(hull_points, hull_points)
    max_width = np.max(distances)
    return max_width

json_fpath = os.path.join(args.data_dir, 'metadata.json')
render_K = np.array([[711.11127387, 0.0, 255.5],
                     [0.0, 711.11127387, 255.5],
                     [0.0, 0.0, 1.0]])
metadata = {'render_K': render_K.tolist()}
scales = {}

with open(json_fpath, 'r') as f:
    metadata = json.load(f)
max_widths = {}

objects = os.listdir(os.path.join(args.data_dir, 'objects'))
for obj in objects:
    obj_path = os.path.join(args.data_dir, 'objects', obj)
    if not os.path.isdir(obj_path):
        continue
    cad_path_real = os.path.join(obj_path, [f for f in os.listdir(obj_path) if f.endswith('.ply') or f.endswith('.obj')][0])
    print(cad_path_real)

    mesh = trimesh.load(cad_path_real, force='mesh')
    model_points = trimesh.sample.sample_surface(mesh, 2048)[0]
    model_points = model_points.astype(np.float32)
    max_width = maximum_width(model_points)
    print(f"Maximum width for {obj}: {max_width}")

    if max_width < 10:
        max_width *= 1000

    max_widths[obj] = max_width

    scale = get_norm_info(cad_path_real)
    
    # print(scale)
    if scale > 1:
        print("The CAD model is in meter")
        # scale /= 1000
    else:
        scale *=1000

    scales[obj] = str(1 / scale * 2)
    print(f"Scale for {obj}: {scales[obj]}")


metadata['max_widths'] = max_widths
with open('./metadata.json', 'w') as f:
    json.dump(metadata, f)


# metadata['scales'] = scales
# with open(json_fpath, 'w') as f:
#     json.dump(metadata, f)