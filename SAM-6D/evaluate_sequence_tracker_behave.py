import os
import numpy as np
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os.path as osp
import multiprocessing
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
import argparse
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import gorilla
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import torchvision.transforms as tvtf
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation

from ism.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from ism.utils.bbox_utils import CropResizePad
from ism.model.utils import Detections, convert_npz_to_json
from ism.model.loss import Similarity
from ism.utils.inout import load_json, save_json_bop23
from ism.utils.visualization_utils import visualize_masks
from ism.inference_multi_object import init_model

from pem.model import pose_estimation_model
from pem.run_inference_batch import get_templates, gt_ref_2_pem_ref
from pem.utils.data_utils import compute_error
from pem.run_inference_batch import visualize as visualize_pem
from pem.utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    compute_error
)
import cv2

from deep_6dof_tracking.utils.camera import Camera
# from foundation_pose.estimater import FoundationPose
from foundation_pose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundation_pose.Utils import erode_depth, bilateral_filter_depth, depth2xyzmap_batch, make_mesh_tensors, compute_mesh_diameter

rgb_transform = tvtf.Compose([tvtf.ToTensor(),
                        tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def depth2world_batch(depth_images, K, device="cuda"):
    h, w = depth_images.shape[1], depth_images.shape[2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create grid of pixel coordinates
    v, u = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device))
    u = u.unsqueeze(0).expand(depth_images.shape[0], -1, -1)  # Shape: (N, H, W)
    v = v.unsqueeze(0).expand(depth_images.shape[0], -1, -1)  # Shape: (N, H, W)

    # Convert to world coordinates
    Z = depth_images  # Depth in mm
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Stack into (N, H, W, 3) array
    world_coords = torch.stack((X, Y, Z), dim=-1)

    # Filter out invalid points (depth = 0)
    valid_mask = Z > 0  # Mask for valid depth values
    valid_world_coords = [world_coords[i][valid_mask[i]] for i in range(world_coords.shape[0])]

    return valid_world_coords

def project_3d_to_2d_batch(points_3d, K):
    # Convert K to batch shape (batch, 3, 3)
    K_batch = K.unsqueeze(0).expand(points_3d.shape[0], -1, -1)

    # Perform batch matrix multiplication: (batch, 3, 3) @ (batch, 3, N_points) -> (batch, 3, N_points)
    points_2d_hom = torch.bmm(K_batch, points_3d.transpose(1, 2))  # (batch, 3, N_points)

    # Transpose back to (batch, N_points, 3) and convert to non-homogeneous coordinates
    points_2d_hom = points_2d_hom.transpose(1, 2)  # (batch, N_points, 3)
    points_2d = points_2d_hom[:, :, :2] / points_2d_hom[:, :, [2]]  # (batch, N_points, 2)

    return points_2d


def crop_and_resize(image, xyxy, padding=0.15, target_size=(512, 512)):
    x1, y1, x2, y2 = xyxy

    # Calculate padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    # Expand bounding box with padding
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.shape[1], x2 + pad_x)
    y2 = min(image.shape[0], y2 + pad_y)

    # Crop the image
    crop = image[y1:y2, x1:x2]

    # Calculate aspect ratio and pad to make it square
    h, w = crop.shape[:2]
    if h > w:
        pad = (h - w) // 2
        crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        pad = (w - h) // 2
        crop = cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize to target size
    resized_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_crop


def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def gt_ref_2_fp_ref(pose):
    reshaped = pose.reshape(4,4)
    reshaped[0,0] = -reshaped[0,0]
    reshaped[0,1] = -reshaped[0,1]
    reshaped[1,2] = -reshaped[1,2]
    reshaped[2,2] = -reshaped[2,2]
    reshaped[1,3] = -reshaped[1,3]
    reshaped[2,3] = -reshaped[2,3]
    return reshaped

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    if depth.shape[0] == 1082:
        depth = depth[1:-1, :]
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def get_test_data(i, data_path, mask, model_points, cfg, rgb, depth):
    cam_path = os.path.join(data_path, 'camera_behave_resized.json')
    cam_info = json.load(open(cam_path))
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    all_images = []

    depth = depth * cam_info['depth_scale'] / 1000.0
    pts = get_point_cloud_from_depth(depth, K)
    mask = np.array(mask).astype(np.uint8)

    image = rgb.copy()

    mask = np.logical_and(mask > 0, depth > 0)
    if np.sum(mask) > 32:
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
    else:
        print('No object detected')
        return None, None
    mask = mask[y1:y2, x1:x2]
    choose = mask.astype(np.float32).flatten().nonzero()[0]

    # pts
    cloud = pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
    center = np.mean(cloud, axis=0)
    tmp_cloud = cloud - center[None, :]
    flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
    if np.sum(flag) < 4:
        print('No enough points in the object')
        return None, None
    choose = choose[flag]
    cloud = cloud[flag]

    if len(choose) <= cfg.n_sample_observed_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
    choose = choose[choose_idx]
    cloud = cloud[choose_idx]

    # rgb
    rgb = rgb.copy()[y1:y2, x1:x2, :][:,:,::-1]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))
    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

    all_rgb.append(torch.FloatTensor(rgb))
    all_cloud.append(torch.FloatTensor(cloud))
    all_rgb_choose.append(torch.IntTensor(rgb_choose).long())

    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, image


def run_cnos(lock, p_dict, all_images_list, cnos_model, object_names):
    n_images_total = 0
    final_scores_total = np.zeros(len(object_names))
    id_total = 0
    while 1:
        with lock:
            if p_dict['join']:
                break
        
        skip = True
        with lock:
            if len(all_images_list) > 0:
                skip = False
                rgbs = []
                masks = []
                id_total += len(all_images_list)
                nb_images = min(8, len(all_images_list))
                for i in all_images_list[-nb_images:]:
                    rgb = i['rgb']
                    mask = i['mask']
                    rgbs.append(rgb)
                    masks.append(mask)
                all_images_list[:] = []
                n_images_total += nb_images

        if skip:
            time.sleep(0.001)
            continue

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bboxs = [np.array(m.getbbox()).astype(np.float32) for m in masks]
        masks = [(np.array([m])/255).astype(np.float32) for m in masks]
        rgbs = [np.array(r) for r in rgbs]
        # detections_list = []
        # for i in range(len(masks)):
        #     detections_list.append({"masks": torch.from_numpy(masks[i]).to(device),
        #                         "boxes": torch.from_numpy(np.array([bboxs[i]])).to(device)})
        detections = {'masks': torch.from_numpy(np.array(masks)[0]).to(device),
                  'boxes': torch.from_numpy(np.array(bboxs)).to(device)}
            
        detections = Detections(detections)
        
        t0 = time.time()

        query_decriptors, query_appe_descriptors = cnos_model.descriptor_model.forward_batch(np.array(rgbs), detections)
        
        descriptor_time = time.time() - t0
        t1 = time.time()

        semantic_scores, best_templates_idxes = cnos_model.compute_semantic_score_multi_object(query_decriptors)

        semantic_time = time.time() - t1
        t2 = time.time()

        final_scores = semantic_scores.cpu().numpy()
        selected_object_baseline = object_names[np.argmax(final_scores.sum(axis=0))]
        final_scores_total += final_scores.sum(axis=0)
        final_scores_mean = final_scores_total / n_images_total
        selected_object = object_names[np.argmax(final_scores_mean)]

        k = 2
        top_idx = np.argsort(final_scores_mean)[::-1][:k]
        top_scores = final_scores_mean[top_idx]
        top1_ratio = top_scores[0] / top_scores[1]

        final_score_time = time.time() - t2
        total_time = time.time() - t0

        with lock:
            p_dict['detected_object'] = selected_object
            p_dict['selected_object_baseline'] = selected_object_baseline
            p_dict['ratios'].append(top1_ratio)
            p_dict['nb_iter'] += 1
            p_dict['descriptor_time'] += descriptor_time
            p_dict['semantic_time'] += semantic_time
            p_dict['final_score_time'] += final_score_time
            p_dict['total_time_cnos'] += total_time
            p_dict['ids'].append(id_total)


def run_foundation_pose(lock_fp,
                        p_dict_fp, 
                        inference_images, 
                        data_dir,
                        fp_model,
                        K,
                        device,
                        fp_init=False,
                        object_name=None):
    glctx = dr.RasterizeCudaContext()
    meshes_tensor = {}
    mesh_diameters = {}
    tfs_to_center = {}
    meshes = {}
    print('FoundationPose device : ', device)
    while 1:
        with lock_fp:
            if p_dict_fp['join']:
                break

        skip = True
        with lock_fp:
            if len(inference_images) > 0:
                skip = False
                rgb = inference_images[-1]['rgb']
                depth = inference_images[-1]['depth']
                selected_object = p_dict_fp['detected_object']
                inference_images[:] = []
                # mesh = meshes[selected_object]
                last_pose_refiner = p_dict_fp['last_pose_refiner']
                last_pose_FP_only = p_dict_fp['last_pose_FP_only']
        
        if skip:
            time.sleep(0.0001)
            continue

        pose_refiner = None
        pose_fp_only = None
        # FoundationPose
        t10 = time.time()

        zfar = np.inf
        depth_refiner = depth / 1e3
        depth_refiner[(depth<0.1) | (depth>=zfar)] = 0
        depth_refiner = torch.as_tensor(depth_refiner, device=device, dtype=torch.float)
        depth_refiner = erode_depth(depth_refiner, radius=3, device='cuda:0')
        depth_refiner = bilateral_filter_depth(depth_refiner, radius=2, device='cuda:0')

        xyz_map = depth2xyzmap_batch(depth_refiner[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]
        
        if not selected_object in meshes.keys():
            cad_path = os.path.join(data_dir, 'objects', selected_object)
            print(cad_path)
            cad_file = [os.path.join(cad_path, f) for f in os.listdir(cad_path) if f.endswith('.ply') or f.endswith('.obj')][0]
            print(cad_file)
            mesh = trimesh.load(cad_file, force='mesh')
            meshes[selected_object] = mesh

            meshes_tensor[selected_object] = make_mesh_tensors(mesh, device=device)
            mesh_diameters[selected_object] = compute_mesh_diameter(mesh.vertices, n_sample=10000)

            max_xyz = mesh.vertices.max(axis=0)
            min_xyz = mesh.vertices.min(axis=0)
            model_center = (min_xyz+max_xyz)/2

            tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
            tf_to_center[:3,3] = -torch.as_tensor(model_center, device='cuda', dtype=torch.float)
            tfs_to_center[selected_object] = tf_to_center

        mesh = meshes[selected_object]
        mesh_tensor = meshes_tensor[selected_object]
        mesh_diameter = mesh_diameters[selected_object]
        tf_to_center = tfs_to_center[selected_object]
        
        try:
            pose, vis = fp_model.predict(mesh=mesh,
                                    mesh_tensors=mesh_tensor,
                                    rgb=rgb,
                                    depth=depth_refiner,
                                    K=K,
                                    ob_in_cams=last_pose_refiner.reshape(1,4,4),
                                    normal_map=None,
                                    xyz_map=xyz_map,
                                    mesh_diameter=mesh_diameter,
                                    glctx=glctx,
                                    iteration=2,
                                    get_vis=False,)
            
            pose_refiner = (pose@tf_to_center).data.cpu().numpy().reshape(4,4)
            last_pose_refiner = pose_refiner.copy()
        except Exception as e:
            pose_refiner = last_pose_refiner

        if last_pose_FP_only is not None:
            # last_pose_FP_only
            try:
                pose_fp_only, vis = fp_model.predict(mesh=mesh,
                                        mesh_tensors=mesh_tensor,
                                        rgb=rgb,
                                        depth=depth_refiner,
                                        K=K,
                                        ob_in_cams=last_pose_FP_only.reshape(1,4,4),
                                        normal_map=None,
                                        xyz_map=xyz_map,
                                        mesh_diameter=mesh_diameter,
                                        glctx=glctx,
                                        iteration=2,
                                        get_vis=False,)
                pose_fp_only = (pose_fp_only@tf_to_center).data.cpu().numpy().reshape(4,4)
                last_pose_FP_only = pose_fp_only.copy()
            except Exception as e:
                pose_fp_only = last_pose_FP_only.copy()

        with lock_fp:
            p_dict_fp['total_fp_time'] += time.time() - t10
            p_dict_fp['wait_fp'] = False

            p_dict_fp['pose_refiner'] = pose_refiner
            p_dict_fp['pose_fp_only'] = pose_fp_only
            p_dict_fp['last_pose_FP_only'] = last_pose_FP_only
            p_dict_fp['last_pose_refiner'] = last_pose_refiner


def run_inference(data_dir, output_dir, seq_dir, pem_cfg, fp_init=False):

    TEST_CNOS = True
    TEST_THREADING = False

    os.makedirs(output_dir, exist_ok=True)

    if not TEST_THREADING:
        # Init PEM model
        random.seed(pem_cfg.rd_seed)
        torch.manual_seed(pem_cfg.rd_seed)
        print("=> creating model ...")
        # print("=> model: {}".format(pem_cfg.model_name))
        # MODEL = importlib.import_module(pem_cfg.model_name)
        pem_model = pose_estimation_model.Net(pem_cfg.model)
        pem_model = pem_model.cuda()
        pem_model.eval()
        checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'Pose_Estimation_Model/pem/checkpoints', 'sam-6d-pem-base.pth')
        print(checkpoint)
        gorilla.solver.load_checkpoint(model=pem_model, filename=checkpoint)
        
    if TEST_CNOS:
        # Init ISM model 
        ism_model = init_model(config_path='configs')
        device = ism_model.descriptor_model.model.device
        print(device)
        # Here, need to load all features
        objects_dir = os.path.join(data_dir, "objects")
        objects = os.listdir(objects_dir)
        object_names = [o for o in objects if os.path.isdir(f"{objects_dir}/{o}")]
        descriptors = torch.stack([
            torch.load(f"{objects_dir}/{o}/templates/descriptors.pt", map_location="cpu", weights_only=False).to("cuda")
            for o in object_names
        ])
        appe_descriptors = torch.stack([
            torch.load(f"{objects_dir}/{o}/templates/appe_descriptors.pt", map_location="cpu", weights_only=False).to("cuda")
            for o in object_names
        ])
        # n_templates = descriptors.shape[1]
        # xyz_maps = [[np.load("{}/{}/templates/xyz/{:05d}.npy".format(objects_dir, o, i)) for i in range(n_templates)] for o in object_names]
        # xyz_maps = torch.tensor(np.array(xyz_maps)).to(device)
        ism_model.ref_data = {
            "descriptors": descriptors,
            "appe_descriptors": appe_descriptors,
        }
        
        logging.info("Loaded descriptors")
        logging.info(descriptors.shape)
        logging.info(appe_descriptors.shape)

    weights_dir = '/gel/usr/chren50/source/FoundationPose-private/foundation_pose/weights'
    refiner = PoseRefinePredictor(weights_dir)

    all_model_points = [np.load(os.path.join(objects_dir, o, 'templates/all_pts.npy')) for o in object_names]
    all_tem_feats_global = [torch.load(os.path.join(objects_dir, o, 'templates/all_tem_feats.pth'), weights_only=True).to(device) for o in object_names]
    all_tem_pts_global = [torch.load(os.path.join(objects_dir, o, 'templates/all_tem_pts.pth'), weights_only=True).to(device) for o in object_names]

    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    seq_name = os.path.basename(seq_dir)
    poses_f = os.path.join(seq_dir, '../../poses', seq_name, 'object_fit_all.npz')
    gt_poses = np.load(poses_f, allow_pickle=True)

    images_names = os.listdir(seq_dir)
    images_names.sort()
    images_names.remove('info.json')
    images_files = [os.path.join(seq_dir, p, 'k1.color.jpg') for p in images_names]

    cam_path = os.path.join(data_dir, 'camera_behave_resized.json')
    cam_info = json.load(open(cam_path))
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    # Shared resources
    all_images_list = manager.list()
    p_dict = manager.dict()
    p_dict['join'] = False
    p_dict['detected_object'] = None
    p_dict['ratios'] = manager.list()
    p_dict['ids'] = manager.list()
    p_dict['descriptor_time'] = 0
    p_dict['semantic_time'] = 0
    p_dict['final_score_time'] = 0
    p_dict['total_time_cnos'] = 0
    p_dict['nb_iter'] = 0
    p_cnos = multiprocessing.Process(target=run_cnos, args=(lock, p_dict, all_images_list, ism_model, object_names))
    p_cnos.start()

    lock_fp = multiprocessing.Lock()
    inference_images_list = manager.list()
    p_dict_fp = manager.dict()
    p_dict_fp['join'] = False
    p_dict_fp['detected_object'] = None
    p_dict['selected_object_baseline'] = None
    p_dict_fp['total_fp_time'] = 0
    p_dict_fp['last_pose_FP_only'] = None
    p_dict_fp['last_pose_refiner'] = None
    p_dict_fp['pose_fp_only'] = None
    p_dict_fp['pose_refiner'] = None
    p_dict_fp['wait_fp'] = False
    p_meshes = manager.dict()
    object_name_fp = None
    if fp_init:
        object_name_fp = seq_name.split('_')[2]
        trans = gt_poses['trans'][0]
        rot = gt_poses['angles'][0]
        rot = Rotation.from_rotvec(rot).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot
        transform_matrix[0:3,3] = trans
        reverse_matrix = np.eye(4)
        reverse_matrix[1, 1] = -1
        reverse_matrix[2, 2] = -1
        init_pose = reverse_matrix @ transform_matrix
        init_pose[1:3,:] = -init_pose[1:3,:]
        print('Init pose for FP only')
        print(init_pose)
        p_dict_fp['last_pose_FP_only'] = init_pose.copy().reshape((4,4))
    p_fp = multiprocessing.Process(target=run_foundation_pose,
                                   args=(lock_fp,
                                        p_dict_fp,
                                        inference_images_list,
                                        data_dir,
                                        refiner,
                                        K,
                                        device,
                                        fp_init,
                                        object_name_fp)
                                    )
    p_fp.start()

    last_pose_pem = None
    desync_counter = 0
    pem_consistence_counter = 0
    out_last = None

    total_time = 0
    successes = 0
    successes_baseline = 0
    total_test_data_time = 0
    total_model_time = 0
    total_load_cad_time = 0

    pred_poses = []
    refiner_poses = []
    fp_only_poses = []
    reset_timestamps = []
    selected_object_list = []
    selected_object_baseline_list = []
    error_df = pd.DataFrame(columns=['Translation', 'Rotation'])
    error_df_correct = pd.DataFrame(columns=['Translation', 'Rotation'])
    error_df_correct_filtered = pd.DataFrame(columns=['Translation', 'Rotation'])
    error_df_tracker = pd.DataFrame(columns=['Translation', 'Rotation'])
    error_df_tracker_only = pd.DataFrame(columns=['Translation', 'Rotation'])
    for i in tqdm(range(len(images_files))):
        t0 = time.time()
        t5 = time.time()

        rgb_path = images_files[i]
        rgb = Image.open(rgb_path).convert("RGB")
        depth_path = rgb_path.replace('k1.color.jpg', 'k1.depth.png')
        depth = imageio.v2.imread(depth_path).astype(np.float32)
        mask_path = rgb_path.replace('k1.color.jpg', 'k1.mask.jpg')
        mask = Image.open(mask_path).convert("L")

        selected_object = None
        with lock:
            all_images_list.append({
                "rgb": rgb,
                "mask": mask,
            })

        while 1:
            with lock:
                selected_object = p_dict['detected_object']
                selected_object_baseline = p_dict['selected_object_baseline']
                if selected_object is not None:
                    break
        

        selected_object_list.append(selected_object)
        selected_object_baseline_list.append(selected_object_baseline)

        if selected_object in seq_dir:
            successes += 1
        if selected_object_baseline in seq_dir:
            successes_baseline += 1
             
        total_load_cad_time += time.time() - t5

        rgb = np.array(rgb)
        with lock_fp:
            if p_dict_fp['last_pose_refiner'] is not None:
                inference_images_list.append({
                    "rgb": rgb.copy(),
                    "depth": depth.copy(),
                })
                p_dict_fp['detected_object'] = selected_object
                p_dict_fp['wait_fp'] = True
        
        t6 = time.time()
        
        object_id = object_names.index(selected_object)
        all_tem_pts = all_tem_pts_global[object_id]
        all_tem_feat = all_tem_feats_global[object_id]
        model_points = all_model_points[object_id]

        input_data, image = get_test_data(i, data_dir, mask, model_points, pem_cfg.test_dataset, rgb, depth)
        
        total_test_data_time += time.time() - t6
        t7 = time.time()
        
        if input_data is None:
            out = out_last
        else :
            ninstance = input_data['pts'].size(0)
            # print("=> running model ...")
            with torch.no_grad():
                input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
                input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
                out = pem_model(input_data)
            out_last = out
        
        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] # * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000


        total_model_time += time.time() - t7

        trans = gt_poses['trans'][i]
        rot = gt_poses['angles'][i]
        rot = Rotation.from_rotvec(rot).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot
        transform_matrix[0:3,3] = trans

        reverse_matrix = np.eye(4)
        reverse_matrix[1, 1] = -1
        reverse_matrix[2, 2] = -1
        my_pose = reverse_matrix @ transform_matrix

        rot = pred_rot[0]
        trans = pred_trans[0] / 1000
        pred_pose = np.eye(4)
        pred_pose[:3,:3] = rot
        pred_pose[:3, 3] = trans

        while 1:
            with lock_fp:
                if p_dict_fp['wait_fp'] == True:
                    time.sleep(0.0001)
                    continue
                else:
                    break

        with lock_fp:
            last_pose_refiner = p_dict_fp['last_pose_refiner']
            last_pose_FP_only = p_dict_fp['last_pose_FP_only']
            pose_refiner = p_dict_fp['pose_refiner']
            pose_fp_only = p_dict_fp['pose_fp_only']

        if last_pose_refiner is None:
            last_pose_refiner = pred_pose.copy()
            error_df_tracker = pd.concat([error_df_tracker, pd.DataFrame([[0, 0]], columns=['Translation', 'Rotation'])])
            error_df_tracker_only = pd.concat([error_df_tracker_only, pd.DataFrame([[0, 0]], columns=['Translation', 'Rotation'])])
        else:
            pose_refiner_cp = pose_refiner.copy()
            pose_refiner_cp = gt_ref_2_pem_ref(pose_refiner_cp)
            diff_t2, diff_r2 = compute_error(pose_refiner_cp, my_pose)
            error_df_tracker = pd.concat([error_df_tracker, pd.DataFrame([[diff_t2[0], diff_r2[0]]], columns=['Translation', 'Rotation'])])
        
            if last_pose_FP_only is None:
                error_df_tracker_only = pd.concat([error_df_tracker_only, pd.DataFrame([[diff_t2[0], diff_r2[0]]], columns=['Translation', 'Rotation'])])
            else:
                # FP only now
                pose_fp_only_cp = pose_fp_only.copy()
                pose_fp_only_cp = gt_ref_2_pem_ref(pose_fp_only_cp)
                diff_t3, diff_r3 = compute_error(pose_fp_only_cp, my_pose)
                error_df_tracker_only = pd.concat([error_df_tracker_only, pd.DataFrame([[diff_t3[0], diff_r3[0]]], columns=['Translation', 'Rotation'])])

        if last_pose_pem is None:
            last_pose_pem = pred_pose.copy()

        current_max_width = metadata['max_widths'][selected_object]
        translation_thresh_pem = 0.20 * current_max_width / 1000
        rotation_thresh_pem = 20
        diff_t_pem, diff_r_pem = compute_error(pred_pose, last_pose_pem)
        if diff_t_pem[0] < translation_thresh_pem and diff_r_pem[0] < rotation_thresh_pem:
            pem_consistence_counter += 1
        else:
            pem_consistence_counter = 0
        last_pose_pem = pred_pose.copy()

        n_reset = 20
        n_reset_consist = 10
        translation_thresh = 0.20 * current_max_width / 1000
        rotation_thresh = 20

        if pose_refiner is not None:
            pose_refiner_cp = pose_refiner.copy()
            diff_t_sync, diff_r_sync = compute_error(pred_pose, pose_refiner_cp)
            
            if selected_object_list[-1] != selected_object_list[-2]:
                desync_counter = 0
                print("Desync detected")
                if last_pose_FP_only is None:
                    last_pose_FP_only = pose_refiner.copy()
                last_pose_refiner = pred_pose.copy()
                reset_timestamps.append(i)

            if diff_t_sync[0] < translation_thresh and diff_r_sync[0] < rotation_thresh:
                desync_counter = 0
            else:
                desync_counter += 1
                if desync_counter > n_reset and pem_consistence_counter > n_reset_consist:
                    print("Desync detected")
                    if last_pose_FP_only is None:
                        last_pose_FP_only = pose_refiner.copy()

                    last_pose_refiner = pred_pose.copy()
                    desync_counter = 0
                    reset_timestamps.append(i)
        pred_poses.append(pred_pose.copy())
        if pose_refiner is not None:
            refiner_poses.append(pose_refiner.copy())
        else:
            refiner_poses.append(pred_pose.copy())
        if pose_fp_only is not None:
            fp_only_poses.append(pose_fp_only.copy())
        else:
            if pose_refiner is not None:
                fp_only_poses.append(pose_refiner.copy())
            else:
                fp_only_poses.append(pred_pose.copy())
        pred_pose = gt_ref_2_pem_ref(pred_pose)


        with lock_fp:
            if last_pose_refiner is None:
                p_dict_fp['last_pose_refiner'] = None
            else:
                p_dict_fp['last_pose_refiner'] = last_pose_refiner.copy()
            if last_pose_FP_only is None:
                p_dict_fp['last_pose_FP_only'] = None
            else:
                p_dict_fp['last_pose_FP_only'] = last_pose_FP_only.copy()

        diff_t, diff_r = compute_error(pred_pose, my_pose)
        error_df = pd.concat([error_df, pd.DataFrame([[diff_t[0], diff_r[0]]], columns=['Translation', 'Rotation'])])
        if selected_object in seq_dir:
            error_df_correct = pd.concat([error_df_correct, pd.DataFrame([[diff_t[0], diff_r[0]]], columns=['Translation', 'Rotation'])])
            if diff_t[0] < 0.5 and diff_r[0] < 89:
                error_df_correct_filtered = pd.concat([error_df_correct_filtered, pd.DataFrame([[diff_t[0], diff_r[0]]], columns=['Translation', 'Rotation'])])
    
        total_time += time.time() - t0

    with lock:
        p_dict['join'] = True
    p_cnos.join()
    with lock_fp:
        p_dict_fp['join'] = True
    p_fp.join()

    print("==================================================")
    print("Average time for each step")
    print(f"Test data time: {total_test_data_time/len(images_files)}")
    print(f"PEM Model time: {total_model_time/len(images_files)}")
    with lock_fp:
        total_FP_time = p_dict_fp['total_fp_time']
    print(f"Pose Refiner time: {total_FP_time/(len(images_files)-1)}")
    print(f"Total time: {total_time/len(images_files)}")
    print(f"Total Load CAD time: {total_load_cad_time}")
    print("==================================================")

    with lock:
        descriptor_time = p_dict['descriptor_time']
        semantic_time = p_dict['semantic_time']
        final_score_time = p_dict['final_score_time']
        total_time_cnos = p_dict['total_time_cnos']
        nb_iter = p_dict['nb_iter']
        ratios = p_dict['ratios']
        ids = p_dict['ids']
        print(f"Descriptor time: {descriptor_time/nb_iter}")
        print(f"Semantic time: {semantic_time/nb_iter}")
        print(f"Final score time: {final_score_time/nb_iter}")
        print(f"Total time cnos: {total_time_cnos/nb_iter}")
        print("==================================================")


        print(f"Success rate: {successes}/{len(images_files)}")
        print(f"Mean ratio: {np.mean(np.array(ratios))}")
        print("==================================================")

        out_csv = f'{output_dir}/cnos_ratios.csv'
        with open(out_csv, 'w') as ff:
          ff.write('frame_number,ids,ratios\n')
          for i, (id, ratio) in enumerate(zip(ids, ratios)):
            ff.write(f'{i},{id},{ratio}\n')

        plt.plot(ids, ratios)
        plt.grid()
        plt.xlabel("Frame #")
        plt.ylabel("top1/top2 Ratio")
        plt.title("CNOS top1/top2 Ratio")
        plt.savefig(f"{output_dir}/ratios.png")

    print(f"Mean translation error: {error_df['Translation'].mean()*1000}")
    print(f"Mean rotation error: {error_df['Rotation'].mean()}")
    error_df.to_csv(f"{output_dir}/pose_error.csv", index=False)
    print("==================================================")
    print(f"Mean translation error for correct detection: {error_df_correct['Translation'].mean()*1000}")
    print(f"Mean rotation error for correct detection: {error_df_correct['Rotation'].mean()}")
    print(f"{len(error_df_correct)} / {len(images_files)}")
    error_df_correct.to_csv(f"{output_dir}/pose_error_correct.csv", index=False)
    print("==================================================")
    print(f"Mean translation error for correct detection and filtered: {error_df_correct_filtered['Translation'].mean()*1000}")
    print(f"Mean rotation error for correct detection and filtered: {error_df_correct_filtered['Rotation'].mean()}")
    print(f"{len(error_df_correct_filtered)} / {len(images_files)}")
    error_df_correct_filtered.to_csv(f"{output_dir}/pose_error_correct_filtered.csv", index=False)
    print("==================================================")
    print(f"Mean translation error for tracker: {error_df_tracker['Translation'].mean()*1000}")
    print(f"Mean rotation error for tracker: {error_df_tracker['Rotation'].mean()}")
    error_df_tracker.to_csv(f"{output_dir}/pose_error_tracker.csv", index=False)
    print("==================================================")
    print(f"Mean translation error for tracker only: {error_df_tracker_only['Translation'].mean()*1000}")
    print(f"Mean rotation error for tracker only: {error_df_tracker_only['Rotation'].mean()}")
    error_df_tracker_only.to_csv(f"{output_dir}/pose_error_tracker_only.csv", index=False)

    pred_poses = np.array(pred_poses)
    np.save(os.path.join(output_dir, "pem_poses.npy"), pred_poses)
    refiner_poses = np.array(refiner_poses)
    np.save(os.path.join(output_dir, "refiner_poses.npy"), refiner_poses)
    fp_only_poses = np.array(fp_only_poses)
    np.save(os.path.join(output_dir, "fp_only_poses.npy"), fp_only_poses)
    resets_df = pd.DataFrame(reset_timestamps, columns=['Frame #'])
    resets_df.to_csv(f"{output_dir}/resets.csv", index=False)

    with open(os.path.join(output_dir, "selected_objects.txt"), 'w') as f:
        for obj in selected_object_list:
            f.write(f"{obj}\n")

    
    with open(os.path.join(output_dir, "selected_objects_baseline.txt"), 'w') as f:
        for obj in selected_object_baseline_list:
            f.write(f"{obj}\n")

    with open(os.path.join(output_dir, "stats_memoire.txt"), 'w') as f:
        f.write(f"Success rate Ours : \n")
        f.write(f"{successes}/{len(images_files)}\n")
        f.write(f"{np.mean(np.array(ratios))}\n")
        f.write(f"{successes/len(images_files) * 100:.2f}\n")
        f.write("===================================================\n")
        f.write(f"Success rate baseline : \n")
        f.write(f"{successes_baseline}/{len(images_files)}\n")
        f.write(f"{successes_baseline/len(images_files) * 100:.2f}\n")
        f.write("===================================================\n")

    # Compare error_df and error_df_tracker
    plt.clf()
    plt.plot(range(len(images_files)), error_df['Translation']*1000, label='PEM')
    plt.plot(range(len(images_files)), error_df_tracker['Translation']*1000, label='Tracker')
    plt.plot(range(len(images_files)), error_df_tracker_only['Translation']*1000, label='Tracker only')
    for i in reset_timestamps:
        plt.axvline(x=i, color='r', linestyle='--')
    plt.xlabel('Frame #')
    plt.ylabel('Translation error (mm)')
    plt.title('Translation error comparison')
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/translation_error.png")
    plt.clf()
    plt.plot(range(len(images_files)), error_df['Rotation'], label='PEM')
    plt.plot(range(len(images_files)), error_df_tracker['Rotation'], label='Tracker')
    plt.plot(range(len(images_files)), error_df_tracker_only['Rotation'], label='Tracker only')
    for i in reset_timestamps:
        plt.axvline(x=i, color='r', linestyle='--')
    plt.xlabel('Frame #')
    plt.ylabel('Rotation error (degree)')
    plt.title('Rotation error comparison')
    # plt.ylim(0, 50)
    plt.legend()
    plt.savefig(f"{output_dir}/rotation_error.png")

def init_pem(args):
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_iter = args.iter

    # cfg.output_dir = args.output_dir
    # cfg.cad_path = args.cad_path
    # cfg.data_path = args.data_dir
    # cfg.seg_path = args.seg_path

    # cfg.det_score_thresh = args.det_score_thresh
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", nargs="?", help="Path to data to run inference on (my organization)")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--seq_path", nargs="?", help="Path to the sequence to run inference on")
    # pem
    parser.add_argument("--gpus", type=str, default="0", help="path to pretrain model")
    parser.add_argument("--model", type=str, default="pose_estimation_model", help="path to model file")
    parser.add_argument("--config", type=str, default="Pose_Estimation_Model/pem/config/base.yaml", help="path to config file, different config.yaml use different config")
    parser.add_argument("--iter", type=int, default=600000, help="epoch num. for testing")
    parser.add_argument("--exp_id", type=int, default=0, help="")
    parser.add_argument('--fp_init', help="Use gt pose and id for FPonly init", action="store_true")
    args = parser.parse_args()
    pem_cfg = init_pem(args)

    run_inference(args.data_dir, args.output_dir, args.seq_path, pem_cfg, fp_init=args.fp_init)