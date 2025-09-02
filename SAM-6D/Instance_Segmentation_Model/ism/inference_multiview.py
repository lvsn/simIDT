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

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23
from utils.visualization_utils import visualize_masks
from tqdm import tqdm

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
    print('depth.shape')
    print(depth.shape)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    print('batch["depth"].shape')
    print(batch["depth"].shape)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(data_dir, output_dir):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='descriptor_only.yaml')

    
    use_appe = True
    use_geo = False

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    logging.info(f"Initalized model: {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    logging.info(f"Moving models to {device} done!")

    # Here, need to load all features
    objects_dir = os.path.join(data_dir, "objects")
    objects = os.listdir(objects_dir)
    object_names = [o for o in objects if os.path.isdir(f"{objects_dir}/{o}")]
    descriptors = torch.stack([
        torch.load(f"{objects_dir}/{o}/templates/descriptors.pt", map_location="cpu").to("cuda")
        for o in object_names
    ])
    appe_descriptors = None
    if use_appe:
        appe_descriptors = torch.stack([
            torch.load(f"{objects_dir}/{o}/templates/appe_descriptors.pt", map_location="cpu").to("cuda")
            for o in object_names
        ])
    # n_templates = descriptors.shape[1]
    # xyz_maps = [[np.load("{}/{}/templates/xyz/{:05d}.npy".format(objects_dir, o, i)) for i in range(n_templates)] for o in object_names]
    # xyz_maps = torch.tensor(np.array(xyz_maps)).to(device)
    model.ref_data = {
        "descriptors": descriptors,
        "appe_descriptors": appe_descriptors,
    }
    logging.info("Loaded descriptors")

    sequences = os.listdir(os.path.join(data_dir, "sequences"))

    descriptor_time = 0
    semantic_time = 0
    final_score_time = 0
    appe_time = 0
    total_time = 0
    successes = 0
    total_best_diff = 0
    failures = []
    best_diffs = []
    for i in range(len(sequences)):
        images_dir = os.path.join(data_dir, "sequences", sequences[i], "rgb")
        images_files = glob.glob(f"{images_dir}/*.png")
        images = [Image.open(f).convert("RGB") for f in images_files]
        images = np.stack([np.array(i) for i in images])
        images_names = [os.path.basename(f) for f in images_files]

        masks_dir = os.path.join(data_dir, "sequences", sequences[i], "masks")
        masks_files = glob.glob(f"{masks_dir}/*.png")
        masks = [Image.open(f).convert("L") for f in masks_files]
        bboxes = np.stack([np.array(m.getbbox()).astype(np.float32) for m in masks])
        masks = np.stack([(np.array(m)/255).astype(np.float32) for m in masks])
        detections = {'masks': torch.from_numpy(masks).to(device),
                      'boxes': torch.from_numpy(bboxes).to(device)}
        logging.info("Detections")
        logging.info(detections['masks'].shape)
        logging.info(detections['boxes'].shape)
        detections = Detections(detections)

        t0 = time.time()
        query_decriptors, query_appe_descriptors = model.descriptor_model.forward_batch(images, detections)
        descriptor_time += time.time() - t0

        t1 = time.time()
        semantic_scores, best_templates_idxes = model.compute_semantic_score_multi_object(query_decriptors)
        semantic_time += time.time() - t1

        t2 = time.time()
        if use_appe:
            # compute the appearance score
            appe_scores, ref_aux_descriptor= model.compute_appearance_score_multi_view(best_templates_idxes, query_appe_descriptors)
        appe_time += time.time() - t2
        
        t4 = time.time()
        
        only_appe = False
        print('shapes')
        print(semantic_scores.shape)
        print(appe_scores.shape)
        appe_scores = appe_scores.squeeze(2)
        final_scores = semantic_scores
        if use_appe:
            final_scores = (final_scores + appe_scores) / 2
        if only_appe:
            final_scores = appe_scores

        
        aggregation_function = "avg_5"
        if aggregation_function == "max":
            final_scores = final_scores.max(dim=0).values
        elif aggregation_function == "mean":
            final_scores = final_scores.mean(dim=0)
        elif aggregation_function == "median":
            final_scores = final_scores.median(dim=0).values
        elif aggregation_function == "avg_5":
            final_scores = final_scores.topk(5, dim=0).values.mean(dim=0)

        final_scores = final_scores.cpu().numpy()
        best_templates_idxes = best_templates_idxes.cpu().numpy()[0]
        selected_object = object_names[np.argmax(final_scores)]
        print("final_scores")
        print(final_scores)
        print("==================================================")
        print('selected object : ', selected_object)
        print('for sequence : ', sequences[i])
        print('==================================================')
        if selected_object in sequences[i]:
            successes += 1
        else:
            failures.append(sequences[i])

        final_score_time += time.time() - t4
        total_time += time.time() - t0

        k = 5
        top_idx = np.argsort(final_scores)[::-1][:k]
        top_scores = final_scores[top_idx]
        best_diff = top_scores[0] - top_scores[1]
        total_best_diff += best_diff
        best_diffs.append(best_diff.item())


    print("==================================================")
    print("Average time for each step")
    print(f"Descriptor time:  {descriptor_time/len(sequences):.3f}")
    print(f"Semantic time:    {semantic_time/len(sequences):.3f}")
    print(f"Appearance time: {appe_time/len(images):.3f}")
    # print(f"Geometric time: {geo_time/len(images):.3f}")
    print(f"Final score time: {final_score_time/len(sequences):.3f}")
    print(f"Total time:       {total_time/len(sequences):.3f}")
    print(f"Success rate:     {successes}/{len(sequences)}")
    print(f"Mean best diff:   {total_best_diff/len(sequences):.3f}")
    print(f"Min best diff:    {min(best_diffs):.3f}")
    print(f"Failures:         {failures}")
    print(f"Best diffs:       {best_diffs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", nargs="?", help="Path to data to run inference on (my organization)")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    args = parser.parse_args()
    run_inference(args.data_dir, args.output_dir)