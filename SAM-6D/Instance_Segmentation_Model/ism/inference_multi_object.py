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
import hydra

from hydra.utils import get_original_cwd, to_absolute_path

from ism.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from ism.utils.bbox_utils import CropResizePad
from ism.model.utils import Detections, convert_npz_to_json
from ism.model.loss import Similarity
from ism.utils.inout import load_json, save_json_bop23
from ism.utils.visualization_utils import visualize_masks
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

def init_model(config_path="configs"):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name='descriptor_only.yaml')

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    logging.info(f"Initalized model: {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    logging.info(f"Moving models to {device} done!")
    return model

def run_inference(data_dir, output_dir):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='descriptor_only.yaml')

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
    logging.info(descriptors.shape)
    logging.info(appe_descriptors.shape)

    images_names = os.listdir(os.path.join(data_dir, "rgb"))
    images_files = [os.path.join(data_dir, "rgb", p) for p in images_names]
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

    
    load_json_time = 0
    load_masks_time = 0
    depth2world_time = 0
    sample_time = 0
    translate_time = 0
    project_time = 0
    geo_score_time = 0
    
    failures = []
    for i in tqdm(range(len(images))):
        t0 = time.time()

        detections = Detections(detections_list[i])
        query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(images[i]), detections)

        descriptor_time += time.time() - t0
        t1 = time.time()

        logging.info("Compute metrics")
        # matching descriptors
        semantic_scores, best_templates_idxes = model.compute_semantic_score_multi_object(query_decriptors)
        semantic_time += time.time() - t1
        t2 = time.time()

        use_appe = True
        
        if use_appe:
            # compute the appearance score
            appe_scores, ref_aux_descriptor= model.compute_appearance_score_multi_object(best_templates_idxes, query_appe_descriptors)

        appe_time += time.time() - t2
        t3 = time.time()

        use_geo = False
        if use_geo:
            t10 = time.time()

            # Load metadata
            metadata = load_json(f"{data_dir}/metadata.json")
            render_K = np.array(metadata['render_K'])

            load_json_time += time.time() - t10
            t11 = time.time()

            # Load mask files
            m_files = [os.path.join(data_dir, 'objects', o, 'templates/masks', '{:05d}.png'.format(i)) 
                    for o, i in zip(object_names, best_templates_idxes[0].cpu().numpy())]

            # Load all masks into a single batch
            masks = [np.array(Image.open(m).convert("L")).astype(np.float32) for m in m_files]
            masks = np.stack(masks)  # Shape: (N, H, W)
            masks = torch.from_numpy(masks).float().to(device) / 255  # Move to GPU

            load_masks_time += time.time() - t11
            t12 = time.time()
 
            # Apply scaling
            scales = torch.tensor([float(metadata['scales'][o]) for o in object_names], device=device)
            masks = masks * scales.view(-1, 1, 1) * 1000  # Scale masks to mm
            points_list = depth2world_batch(masks, render_K)

            depth2world_time += time.time() - t12
            t13 = time.time()

            idx = [torch.randint(0, points.shape[0], size=(2048,)) for points in points_list]
            points = torch.stack([points[i] for points, i in zip(points_list, idx)])

            sample_time += time.time() - t13
            t14 = time.time()

            adjust = torch.zeros_like(points)
            adjust[:, :, 2] = -scales.view(-1, 1) * 1000
            points = points + adjust

            batch = batch_input_data(images_files[i].replace('rgb', 'depth'), os.path.join(data_dir, 'camera.json'), device)
            translate = model.Calculate_the_query_translation(detections.masks[0], batch["depth"][0], batch["cam_intrinsic"][0], batch['depth_scale'])
            points = points/1000 + translate.view(1, 1, 3)

            translate_time += time.time() - t14
            t15 = time.time()

            K = np.array(json.load(open(os.path.join(data_dir, 'camera.json')))['cam_K']).reshape((3, 3))
            K = torch.from_numpy(K).to(device).float()
            points_2d = project_3d_to_2d_batch(points, K)

            project_time += time.time() - t15
            t16 = time.time()

            geo_score, visible_ratio = model.compute_geometric_score(points_2d, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred)

            geo_score_time += time.time() - t16

        geo_time += time.time() - t3
        t4 = time.time()

        denom = 1
        final_scores = semantic_scores
        if use_appe:
            final_scores += appe_scores
            denom += 1
        if use_geo:
            final_scores += geo_score*visible_ratio
            denom += visible_ratio
        final_scores /= denom

        final_scores = final_scores.cpu().numpy()[0]
        best_templates_idxes = best_templates_idxes.cpu().numpy()[0]
        selected_object = object_names[np.argmax(final_scores)]
        print("final_scores")
        print(final_scores)
        print("==================================================")
        print('selected object : ', selected_object)
        print('for scene : ', images_names[i])
        print('==================================================')
        if selected_object in images_names[i]:
            successes += 1
        else:
            failures.append(images_names[i])

        final_score_time += time.time() - t4
        total_time += time.time() - t0


        final_score = torch.tensor(final_scores.max())
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()
    
        save_path = f"{output_dir}/sam6d_results/detection_ism"
        detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", detections)
        # vis_img = visualize(np.array(images[i]), detections, f"{output_dir}/sam6d_results/vis_ism.png")
        # vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")


        visualization = False
        if visualization:
            # Visualization
            k = 5
            top_idx = np.argsort(final_scores)[::-1][:k]
            top_scores = final_scores[top_idx]
            top_obj = [object_names[j] for j in top_idx]
            top_templates = [best_templates_idxes[j] for j in top_idx]

            prop = crop_and_resize(np.array(images[i]), detections.boxes[0].cpu().numpy(), padding=0.15, target_size=(512, 512))

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            best_diff = round(top_scores[0] - top_scores[1], 4)
            total_best_diff += best_diff
            axs[0, 0].imshow(prop)
            axs[0, 0].set_title("Input : {:.3f}".format(best_diff))

            for j in range(1, k+1):
                template_img = Image.open("{}/{}/templates/rgb/{:05d}.png".format(objects_dir, top_obj[j-1], top_templates[j-1]))
                axs[j//3, j%3].imshow(template_img)
                rounded = round(top_scores[j-1], 2)
                axs[j//3, j%3].set_title("{}: {:.2f}".format(top_obj[j-1], rounded))

            plt.savefig(f"{output_dir}/{images_names[i]}")
            if not selected_object in images_names[i]:
                os.makedirs(f"{output_dir}/failures", exist_ok=True)
                plt.savefig(f"{output_dir}/failures/{images_names[i]}")

    print("==================================================")
    print("Average time for each step")
    print(f"Descriptor time: {descriptor_time/len(images)}")
    print(f"Semantic time: {semantic_time/len(images)}")
    print(f"Appearance time: {appe_time/len(images)}")
    print(f"Geometric time: {geo_time/len(images)}")
    print(f"Final score time: {final_score_time/len(images)}")
    print(f"Total time: {total_time/len(images)}")
    print(f"Success rate: {successes}/{len(images)}")
    print(f"Mean best diff: {total_best_diff/len(images)}")
    print(f"Failures: {failures}")

    print("==================================================")
    print("Average time for each step")
    print(f"Load json time: {load_json_time/len(images)}")
    print(f"Load masks time: {load_masks_time/len(images)}")
    print(f"Depth2world time: {depth2world_time/len(images)}")
    print(f"Sample time: {sample_time/len(images)}")
    print(f"Translate time: {translate_time/len(images)}")
    print(f"Project time: {project_time/len(images)}")
    print(f"Geo score time: {geo_score_time/len(images)}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", nargs="?", help="Path to data to run inference on (my organization)")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    args = parser.parse_args()
    run_inference(args.data_dir, args.output_dir)