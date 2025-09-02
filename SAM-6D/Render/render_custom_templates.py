import blenderproc as bproc
import bpy

import os
import argparse
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
args = parser.parse_args()

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/ism/utils/poses/predefined_poses/cam_poses_level0.npy')

bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)


# load cnos camera pose
cam_poses = np.load(cnos_cam_fpath)

# calculating the scale of CAD model
cad_path_real = os.path.join(args.cad_path, [f for f in os.listdir(args.cad_path) if f.endswith('.ply') or f.endswith('.obj')][0])
print(cad_path_real)
is_in_meter = False
if args.normalize:
    scale = get_norm_info(cad_path_real)
else:
    scale = 1

print(scale)
if scale > 1:
    print("The CAD model is in meter")
    is_in_meter = True

# check save folder
save_fpath = os.path.join(args.cad_path, "templates")
os.makedirs(save_fpath, exist_ok=True)
os.makedirs(os.path.join(save_fpath, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(save_fpath, 'masks'), exist_ok=True)
os.makedirs(os.path.join(save_fpath, 'xyz'), exist_ok=True)

# # Remove all materials
# for material in bpy.data.materials:
#     bpy.data.materials.remove(material)

for idx, cam_pose in enumerate(cam_poses):

    print("Rendering template: ", idx)
    
    bproc.clean_up()

    # load object
    obj = bproc.loader.load_obj(cad_path_real)[0]

    for material in bpy.data.materials:
        if material.users == 0:  # Check if the material is unused
            bpy.data.materials.remove(material)

    obj.set_scale([scale, scale, scale])
    
    obj.set_cp("category_id", 1)

    # assigning material colors to untextured objects
    # if args.colorize:
    #     color = [args.base_color, args.base_color, args.base_color, 0.]
    #     material = bproc.material.create('obj')
    #     material.set_principled_shader_value('Base Color', color)
    #     obj.set_material(0, material)

    # convert cnos camera poses to blender camera poses
    print(cam_pose)
    print(np.linalg.inv(cam_pose))
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]

    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2

    bproc.camera.add_camera_pose(cam_pose)
    
    # set light
    light_scale = 2.5
    light_energy = 1000
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    light1.set_energy(light_energy)

    for material in bpy.data.materials:
        if material.users == 0:  # Check if the material is unused
            bpy.data.materials.remove(material)

    bproc.renderer.set_max_amount_of_samples(50)
    # render the whole pipeline
    data = bproc.renderer.render()
    # render nocs
    data.update(bproc.renderer.render_nocs())
    

    # save rgb image
    color_bgr_0 = data["colors"][0]
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
    # cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)
    cv2.imwrite(os.path.join(save_fpath,'rgb', '{:05d}.png'.format(idx)), color_bgr_0)

    # save mask
    mask_0 = data["nocs"][0][..., -1]
    # cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
    cv2.imwrite(os.path.join(save_fpath,'masks', '{:05d}.png'.format(idx)), mask_0*255)
    
    # save nocs
    xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
    if is_in_meter:
        xyz_0 = xyz_0 * 1000
        xyz_0[xyz_0[:,:,-1] < -999] = [-1, -1, -1]
    # np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))
    np.save(os.path.join(save_fpath,'xyz', '{:05d}.npy'.format(idx)), xyz_0.astype(np.float16))