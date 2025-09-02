import os, sys
import numpy as np
import json
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import gorilla
import importlib
import random
import pandas as pd
from tqdm import tqdm

from ism.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from ism.utils.bbox_utils import CropResizePad
from ism.model.utils import Detections, convert_npz_to_json
from ism.model.loss import Similarity
from ism.utils.inout import load_json, save_json_bop23
from ism.utils.visualization_utils import visualize_masks
from ism.inference_multi_object import init_model

from pem.model import pose_estimation_model
from pem.run_inference_batch import get_templates, get_test_data, gt_ref_2_pem_ref
from pem.utils.data_utils import compute_error
from pem.run_inference_batch import visualize as visualize_pem



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

def run_inference(data_dir, output_dir, seq_dir, pem_cfg):
    # Init ISM model 
    ism_model = init_model(config_path='configs')
    device = ism_model.descriptor_model.model.device
    print(device)

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

    templates = {}
    template_points = {}
    template_choose = {}
    template_features = {}
    model_points_all = {}

    poses_f = os.path.join(seq_dir, 'poses.npy')
    gt_poses = np.load(poses_f)

    # Here, need to load all features
    objects_dir = os.path.join(data_dir, "objects")
    objects = os.listdir(objects_dir)
    object_names = [o for o in objects if os.path.isdir(f"{objects_dir}/{o}")]
    descriptors = torch.stack([
        torch.load(f"{objects_dir}/{o}/templates/descriptors.pt", map_location="cpu").to("cuda")
        for o in object_names
    ])
    appe_descriptors = torch.stack([
        torch.load(f"{objects_dir}/{o}/templates/appe_descriptors.pt", map_location="cpu").to("cuda")
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

    images_names = os.listdir(os.path.join(seq_dir, "rgb"))
    images_names.sort()
    print(images_names)
    print(images_names.sort())
    images_files = [os.path.join(seq_dir, "rgb", p) for p in images_names]
    images = [Image.open(p).convert("RGB") for p in images_files]

    masks_files = [i.replace("rgb", "masks") for i in images_files]
    detections_list = []
    for m in masks_files:
        mask = Image.open(m).convert("L")
        bbox_xywh = np.array(mask.getbbox()).astype(np.float32)
        mask = (np.array([mask])/255).astype(np.float32)
        detections_list.append({"masks": torch.from_numpy(mask).to(device),
                                "boxes": torch.from_numpy(np.array([bbox_xywh])).to(device)})
    
    descriptor_time = 0
    semantic_time = 0
    appe_time = 0
    geo_time = 0
    final_score_time = 0
    total_time = 0
    successes = 0
    total_best_diff = 0
    total_test_data_time = 0
    total_model_time = 0
    total_load_cad_time = 0

    failures = []
    ratios = []
    pred_poses = []    
    error_df = pd.DataFrame(columns=['Translation', 'Rotation'])
    error_df_correct = pd.DataFrame(columns=['Translation', 'Rotation'])
    error_df_correct_filtered = pd.DataFrame(columns=['Translation', 'Rotation'])
    for i in tqdm(range(len(images))):
        t0 = time.time()

        detections = Detections(detections_list[i])
        query_decriptors, query_appe_descriptors = ism_model.descriptor_model.forward(np.array(images[i]), detections)

        descriptor_time += time.time() - t0
        t1 = time.time()

        logging.info("Compute metrics")
        # matching descriptors
        semantic_scores, best_templates_idxes = ism_model.compute_semantic_score_multi_object(query_decriptors)
        semantic_time += time.time() - t1
        t2 = time.time()
        appe_time += time.time() - t2
        t3 = time.time()
        geo_time += time.time() - t3
        t4 = time.time()

        denom = 1
        final_scores = semantic_scores
        final_scores /= denom

        final_scores = final_scores.cpu().numpy()[0]
        best_templates_idxes = best_templates_idxes.cpu().numpy()[0]
        selected_object = object_names[np.argmax(final_scores)]
        print("final_scores")
        # print(final_scores)
        print("==================================================")
        print('selected object : ', selected_object)
        print('for scene : ', images_names[i])
        print('==================================================')
        if selected_object in seq_dir:
            successes += 1
        else:
            failures.append(images_names[i])

        final_score_time += time.time() - t4

        final_score = torch.tensor(final_scores.max())
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()

        k = 5
        top_idx = np.argsort(final_scores)[::-1][:k]
        top_scores = final_scores[top_idx]
        ratio = top_scores[0] / top_scores[1]
        ratios.append(ratio)

        t5 = time.time()

        # Pose Estimation
        # If there is a new detection, load the templates
        if not selected_object in templates.keys():
            print(f"Loading templates for {selected_object}")
            cad_path = os.path.join(data_dir, 'objects', selected_object)
            tem_path = os.path.join(cad_path, 'templates')
            all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, pem_cfg.test_dataset)
            with torch.no_grad():
                all_tem_pts, all_tem_feat = pem_model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)
            templates[selected_object] = all_tem
            template_points[selected_object] = all_tem_pts
            template_choose[selected_object] = all_tem_choose
            template_features[selected_object] = all_tem_feat

            cad_path_real = os.path.join(cad_path, [f for f in os.listdir(cad_path) if f.endswith('.ply') or f.endswith('.obj')][0])
            mesh : trimesh.Trimesh = trimesh.load(cad_path_real, force='mesh')
            print(mesh.scale)
            model_points = mesh.sample(pem_cfg.test_dataset.n_sample_model_point).astype(np.float32)
            if mesh.scale > 1:
                print('apply scale')
                model_points = model_points / 1000.0
            radius = np.max(np.linalg.norm(model_points, axis=1))
            print(radius)

            model_points_all[selected_object] = model_points

        total_load_cad_time += time.time() - t5
        t6 = time.time()

        all_tem, all_tem_pts, all_tem_choose, all_tem_feat, model_points = templates[selected_object], template_points[selected_object], template_choose[selected_object], template_features[selected_object], model_points_all[selected_object]

        input_data, image = get_test_data(i, data_dir, seq_dir, model_points, pem_cfg.test_dataset)
        ninstance = input_data['pts'].size(0)

        total_test_data_time += time.time() - t6
        t7 = time.time()

        print("=> running model ...")
        with torch.no_grad():
            input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
            input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
            out = pem_model(input_data)
        
        if 'pred_pose_score' in out.keys():
            # print('pred_pose_score')
            # print(out['pred_pose_score'])
            # print(out['score'])
            pose_scores = out['pred_pose_score'] # * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

        total_model_time += time.time() - t7

        my_pose = gt_poses[i]
        rot = pred_rot[0]
        trans = pred_trans[0] / 1000
        pred_pose = np.eye(4)
        pred_pose[:3,:3] = rot
        pred_pose[:3, 3] = trans
        pred_poses.append(pred_pose)
        pred_pose = gt_ref_2_pem_ref(pred_pose)
        my_pose = my_pose.reshape(4,4)
        diff_t, diff_r = compute_error(pred_pose, my_pose)
        print(f'Translation error: {diff_t}')
        print(f'Rotation error: {diff_r}')
        print(f'Pose score: {pose_scores[0]}')
        error_df = pd.concat([error_df, pd.DataFrame([[diff_t[0], diff_r[0]]], columns=['Translation', 'Rotation'])])
        if selected_object in seq_dir:
            error_df_correct = pd.concat([error_df_correct, pd.DataFrame([[diff_t[0], diff_r[0]]], columns=['Translation', 'Rotation'])])
            if diff_t[0] < 0.5 and diff_r[0] < 89:
                error_df_correct_filtered = pd.concat([error_df_correct_filtered, pd.DataFrame([[diff_t[0], diff_r[0]]], columns=['Translation', 'Rotation'])])
        
        total_time += time.time() - t0


    print("==================================================")
    print("Average time for each step")
    print(f"Descriptor time: {descriptor_time/len(images)}")
    print(f"Semantic time: {semantic_time/len(images)}")
    print(f"Final score time: {final_score_time/len(images)}")
    print(f"Test data time: {total_test_data_time/len(images)}")
    print(f"PEM Model time: {total_model_time/len(images)}")
    print(f"Total time: {total_time/len(images)}")
    print(f"Total Load CAD time: {total_load_cad_time}")
    print("==================================================")

    print(f"Success rate: {successes}/{len(images)}")
    print(f"Mean ratio: {np.mean(np.array(ratios))}")
    print("==================================================")

    print(f"Mean translation error: {error_df['Translation'].mean()*1000}")
    print(f"Mean rotation error: {error_df['Rotation'].mean()}")
    print("==================================================")
    print(f"Mean translation error for correct detection: {error_df_correct['Translation'].mean()*1000}")
    print(f"Mean rotation error for correct detection: {error_df_correct['Rotation'].mean()}")
    print(f"{len(error_df_correct)} / {len(images)}")
    print("==================================================")
    print(f"Mean translation error for correct detection and filtered: {error_df_correct_filtered['Translation'].mean()*1000}")
    print(f"Mean rotation error for correct detection and filtered: {error_df_correct_filtered['Rotation'].mean()}")
    print(f"{len(error_df_correct_filtered)} / {len(images)}")


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

    cfg.output_dir = args.output_dir
    # cfg.cad_path = args.cad_path
    cfg.data_path = args.data_dir
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

    args = parser.parse_args()
    pem_cfg = init_pem(args)
    run_inference(args.data_dir, args.output_dir, args.seq_path, pem_cfg)