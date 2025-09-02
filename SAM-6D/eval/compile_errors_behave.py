import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import re
from pathlib import Path
from tqdm import tqdm

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.model_rend_test import ModelRenderer3
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import compute_2Dboundingbox
from deep_6dof_tracking.data.utils import compute_axis, image_blend
from deep_6dof_tracking.eccv.eval_functions import compute_pose_diff, get_pose_difference

def draw_debug(img, pose, camera, alpha, img_render, bb):
    img_render = cv2.resize(img_render, (bb[2, 1] - bb[0, 1], bb[1, 0] - bb[0, 0]))
    bb_copy = bb.copy()
    bb[bb<0] = 0
    crop = img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :]

    h, w, c = crop.shape
    x_min = bb_copy[0, 1]*-1 if bb_copy[0, 1] < 0 else 0
    y_min = bb_copy[0, 0]*-1 if bb_copy[0, 0] < 0 else 0
    x_max = min(w, bb_copy[2, 1])
    y_max = min(h, bb_copy[1, 0])
    
    img_render = img_render[y_min:, x_min:, :]
    blend = image_blend(img_render[:h, :w, ::-1], crop)
    try:
        bb[bb<0] = 0
        img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :] = cv2.addWeighted(img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :],
                                                                    1 - alpha, blend, alpha, 1)
    except Exception as e:
        print(e)

def draw_axis(pose, camera, img, alt_color=False):
    axis = compute_axis(pose, camera)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] if not alt_color else [(255, 255, 0), (255, 0, 255), (0, 255, 255)]
    cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), colors[0], 3)
    cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), colors[1], 3)
    cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), colors[2], 3)

    # cv2.circle(img, tuple(axis[0, ::-1]), 5, colors[0], -1)

def extract_object_name(path_str):
    # Get last part of the path
    folder = Path(path_str).name
    
    # Match the pattern: DateXX_SubYY_objectname_suffix
    match = re.match(r"Date\d+_Sub\d+_(.+?)_", folder)
    if match:
        return match.group(1)
    else:
        print(folder)
        name = folder.split('_')[2]
        return name

def gt_ref_2_pem_ref(pose):
  pose[1,:] = -pose[1,:]
  pose[2,:] = -pose[2,:]
  return pose

def plot_cnos_ratios(exp_dir):
    experiments = os.listdir(args.dir)
    
    max_x = 0
    min_y = 10000
    max_y = 0
    # Create figure for CNOS ratio
    for exp in experiments:
        if not os.path.isdir(os.path.join(args.dir, exp)):
            continue
        if not os.path.exists(os.path.join(args.dir, exp, 'cnos_ratios.csv')):
            print(f"Skipping {exp} as cnos_ratios.csv does not exist.")
            continue
        exp_dir = os.path.join(args.dir, exp)
        print(f"Processing {exp}")
        
        ratio_file = os.path.join(exp_dir, 'cnos_ratios.csv')
        ratios = pd.read_csv(ratio_file)
        plt.plot(ratios['frame_number'], ratios['ratios'], label=exp)
        max_x = max(max_x, ratios['frame_number'].max())
        max_y = max(max_y, ratios['ratios'].max())
        min_y = min(min_y, ratios['ratios'].min())


    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('Top1/Top2 Ratio')
    plt.xlim(0, max_x + 100)
    plt.ylim(min_y - 0.1, max_y + 0.3)
    plt.title('CNOS Ratios')
    plt.grid()
    plt.savefig(os.path.join(args.dir, 'cnos_ratios.png'))

def compile_ratios(exp_dir):
    experiments = os.listdir(exp_dir)
    total_frames = 0
    total_top1 = 0
    total_top1_baseline = 0
    total_ratio = 0
    for exp in experiments:
        if not os.path.isdir(os.path.join(exp_dir, exp)):
            continue
        if not os.path.exists(os.path.join(exp_dir, exp, 'cnos_ratios.csv')):
            print(f"Skipping {exp} as cnos_ratios.csv does not exist.")
            continue
        exp_dir2 = os.path.join(exp_dir, exp)
        print(f"Processing {exp}")
        
        ratio_file = os.path.join(exp_dir2, 'cnos_ratios.csv')
        ratios = pd.read_csv(ratio_file)


        ids_path = os.path.join(exp_dir2, 'selected_objects.txt')
        with open(ids_path, 'r') as f:
            ids = f.read().splitlines()
        ids_path_baseline = os.path.join(exp_dir2, 'selected_objects_baseline.txt')
        with open(ids_path_baseline, 'r') as f:
            ids_baseline = f.read().splitlines()
        
        object_name = extract_object_name(exp_dir2)
        print(object_name)

        counter_top1 = 0
        counter_ratio = 0
        for i in range(len(ids)):
            if ids[i] == object_name:
                try:
                    counter_top1 += 1
                    if ratios['ratios'][i] > 1.1:
                        counter_ratio += 1
                except KeyError:
                    continue

        counter_top1_baseline = 0
        for i in range(len(ids_baseline)):
            if ids_baseline[i] == object_name:
                counter_top1_baseline += 1

        
        out_path = os.path.join(exp_dir2, 'compiled_ratios.txt')
        with open(out_path, 'w') as f:
            f.write(f"Experiment: {exp}\n")
            f.write(f"Top1 count: {counter_top1}\n")
            f.write(f"Top1 percentage: {counter_top1/len(ids)*100:.2f}\n")
            f.write(f"Baseline Top1 count: {counter_top1_baseline}\n")
            f.write(f"Baseline Top1 percentage: {counter_top1_baseline/len(ids_baseline)*100:.2f}\n")
            f.write(f"Total frames: {len(ids)}\n")
            f.write(f"Total frames with ratio > 1.1: {len(ratios[ratios['ratios']>1.1])}\n")
            f.write(f"Total frames with ratio > 1.1 percentage: {len(ratios[ratios['ratios']>1.1])/len(ids)*100:.2f}%\n")
            f.write("\n")

        total_frames += len(ids)
        total_top1 += counter_top1
        total_ratio += counter_ratio
        total_top1_baseline += counter_top1_baseline
    
    with open(os.path.join(exp_dir, 'compiled_ratios.txt'), 'w') as f:
        f.write(f"Overall Top1 count: {total_top1}\n")
        f.write(f"Overall Top1 percentage: {total_top1/total_frames*100:.2f}\n")
        f.write(f"Overall Baseline Top1 count: {total_top1_baseline}\n")
        f.write(f"Overall Baseline Top1 percentage: {total_top1_baseline/total_frames*100:.2f}\n")
        f.write(f"Overall Total frames: {total_frames}\n")
        f.write(f"Overall Total frames with ratio > 1.1: {total_ratio}\n")
        f.write(f"Overall Total frames with ratio > 1.1 percentage: {total_ratio/total_frames*100:.2f}%\n")
        f.write("\n")


def compile_errors(exp_dir):
    experiments = os.listdir(exp_dir)
    
    total_frames = 0
    total_trans_err = 0
    total_rot_err = 0
    total_trans_err_pem = 0
    total_rot_err_pem = 0
    total_trans_err_fp = 0
    total_rot_err_fp = 0
    total_loss = 0
    total_loss_pem = 0
    total_loss_fp = 0
    for exp in experiments:
        if not os.path.isdir(os.path.join(exp_dir, exp)):
            continue
        if not os.path.exists(os.path.join(exp_dir, exp, 'pose_error_tracker.csv')):
            print(f"Skipping {exp}, no pose_error_tracker.csv found.")
            continue
        exp_dir2 = os.path.join(exp_dir, exp)
        err_file = os.path.join(exp_dir2, 'pose_error_tracker.csv')
        df = pd.read_csv(err_file)
        err_file_pem = os.path.join(exp_dir2, 'pose_error.csv')
        df_pem = pd.read_csv(err_file_pem)
        err_file_fp = os.path.join(exp_dir2, 'pose_error_tracker_only.csv')
        df_fp = pd.read_csv(err_file_fp)

        ceiling = 1
        df[df['Translation'] > ceiling] = ceiling
        df_pem[df_pem['Translation'] > ceiling] = ceiling
        df_fp[df_fp['Translation'] > ceiling] = ceiling

        metadata_path = os.path.join('../Data/demo_pem', 'metadata.json')
        metadata = json.load(open(metadata_path))
        if 'Date' in exp:
            obj_name = exp.split('_')[2]
        else:
            obj_name = exp
        width = metadata['max_widths'][obj_name]
        print(f"Processing {exp} with {df.shape[0]} frames, object max width: {width}")

        total_trans_err += df['Translation'].sum()
        total_rot_err += df['Rotation'].sum()
        total_trans_err_pem += df_pem['Translation'].sum()
        total_rot_err_pem += df_pem['Rotation'].sum()
        total_trans_err_fp += df_fp['Translation'].sum()
        total_rot_err_fp += df_fp['Rotation'].sum()
        frame_num = df.shape[0]
        total_frames += frame_num
        
        translation_thresh = 0.25 * width / 1000
        rotation_thresh = 20
        num_loss = df[(df['Translation'] > translation_thresh) | (df['Rotation'] > rotation_thresh)].shape[0]
        num_loss_pem = df_pem[(df_pem['Translation'] > translation_thresh) | (df_pem['Rotation'] > rotation_thresh)].shape[0]
        num_loss_fp = df_fp[(df_fp['Translation'] > translation_thresh) | (df_fp['Rotation'] > rotation_thresh)].shape[0]
        total_loss += num_loss
        total_loss_pem += num_loss_pem
        total_loss_fp += num_loss_fp

        with open(os.path.join(exp_dir2, 'compiled_errors.txt'), 'w') as f:
            f.write(f"Experiment: {exp}\n")
            f.write(f"Total frames: {df.shape[0]}\n")
            f.write(f"Mean Translation Error: {df['Translation'].mean()*1000:.2f}\n")
            f.write(f"Mean Rotation Error: {df['Rotation'].mean():.2f}\n")
            f.write(f"Losses: {num_loss}\n")
            f.write(f"Losses percentage: {num_loss/frame_num*100:.2f}\n")
            f.write(f"Mean Translation Error PEM: {df_pem['Translation'].mean()*1000:.2f}\n")
            f.write(f"Mean Rotation Error PEM: {df_pem['Rotation'].mean():.2f}\n")
            f.write(f"Losses PEM: {num_loss_pem}\n")
            f.write(f"Losses PEM percentage: {num_loss_pem/frame_num*100:.2f}\n")
            f.write(f"Mean Translation Error FP: {df_fp['Translation'].mean()*1000:.2f}\n")
            f.write(f"Mean Rotation Error FP: {df_fp['Rotation'].mean():.2f}\n")
            f.write(f"Losses FP: {num_loss_fp}\n")
            f.write(f"Losses FP percentage: {num_loss_fp/frame_num*100:.2f}\n")
        
        print(f"Losses for {exp}: {num_loss} ({num_loss/frame_num*100:.2f}%)")

        

    with open(os.path.join(exp_dir, 'compiled_errors.txt'), 'w') as f:
        f.write(f"Overall Total frames: {total_frames}\n")
        f.write(f"Overall Mean Translation Error: {total_trans_err/total_frames*1000:.2f}\n")
        f.write(f"Overall Mean Rotation Error: {total_rot_err/total_frames:.2f}\n")
        f.write(f"Overall Losses: {total_loss}\n")
        f.write(f"Overall Losses percentage: {total_loss/total_frames*100:.2f}\n")
        f.write(f"Overall Mean Translation Error PEM: {total_trans_err_pem/total_frames*1000:.2f}\n")
        f.write(f"Overall Mean Rotation Error PEM: {total_rot_err_pem/total_frames:.2f}\n")
        f.write(f"Overall Losses PEM: {total_loss_pem}\n")
        f.write(f"Overall Losses PEM percentage: {total_loss_pem/total_frames*100:.2f}\n")
        f.write(f"Overall Mean Translation Error FP: {total_trans_err_fp/total_frames*1000:.2f}\n")
        f.write(f"Overall Mean Rotation Error FP: {total_rot_err_fp/total_frames:.2f}\n")
        f.write(f"Overall Losses FP: {total_loss_fp}\n")
        f.write(f"Overall Losses FP percentage: {total_loss_fp/total_frames*100:.2f}\n")


def generate_video(exp_dir, data_dir, seq_dir, pose_name='pem'):
    camera_path = os.path.join(data_dir, 'camera_behave.json')
    K = json.load(open(camera_path))
    K = np.array(K['cam_K'])
    K = K/2
    print(K)
    camera = Camera.from_matrix(K.reshape(3, 3), 2048//2, 1536//2)
    print(camera)
    object_name = extract_object_name(exp_dir)
    print(f"Object name: {object_name}")
    cad_path = os.path.join(data_dir, 'objects', object_name)
    texture_path = [os.path.join(cad_path, f) for f in os.listdir(cad_path) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')][0]
    cad_path = [os.path.join(cad_path, f) for f in os.listdir(cad_path) if f.endswith('.obj')][0]
    shader_path = '/gel/usr/chren50/source/deep_6dof_tracking/deep_6dof_tracking/data/shaders'
    renderer = ModelRenderer3(cad_path, shader_path, texture_path, camera, [(174, 174)],)

    max_width = renderer.object_max_width
    bounding_box_width = 1.15 * max_width
    poses = os.path.join(exp_dir, f'{pose_name}_poses.npy')
    poses = np.load(poses)

    frame_names = os.listdir(seq_dir)
    frame_names.sort()
    frame_names.remove('info.json')
    rgb_files = [os.path.join(seq_dir, f, 'k1.color.jpg') for f in frame_names]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(exp_dir, f'video_{pose_name}.mp4'), fourcc, 30, (2048//2, 1536//2))

    for i in tqdm(range(poses.shape[0])):
        pose = poses[i]
        pose = gt_ref_2_pem_ref(pose)
        rgb = cv2.imread(rgb_files[i])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (2048//2, 1536//2))
        pose = Transform.from_matrix(pose)
        bb = compute_2Dboundingbox(pose, camera, bounding_box_width, scale=(1000, 1000, -1000))
        bb2 = compute_2Dboundingbox(pose, camera, bounding_box_width, scale=(1000, -1000, -1000))
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        try:
            renderer.setup_camera(camera, left, right, bottom, top)
            render_rgb, _ = renderer.render_image(pose, ambiant_light=np.array([0.5, 0.5, 0.5]),
                                                                    light_diffuse=np.array([0.4, 0.4, 0.4]))
            
            axis = compute_axis(pose, camera)
            bb2[:,0] -= (bb2[0, 0] + bb2[1, 0])//2 - axis[0, 0]
            bb2[:,1] -= (bb2[0, 1] + bb2[2, 1])//2 - axis[0, 1]
            
            obj_mask = np.zeros_like(render_rgb, dtype=np.uint8)
            obj_mask[render_rgb > 0] = 255
            obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_RGB2GRAY)
            obj_mask_new = cv2.dilate(obj_mask, np.ones((5, 5), np.uint8), iterations=1)
            obj_mask_new[obj_mask > 0] = 0
            obj_mask_new = cv2.dilate(obj_mask_new, np.ones((5, 5), np.uint8), iterations=1)
            obj_mask_color = np.zeros((render_rgb.shape[0], render_rgb.shape[1], 3), dtype=np.uint8)
            obj_mask_color[obj_mask_new > 0] = [0, 255, 0]


            draw_debug(rgb, pose, camera, 0.8, obj_mask_color, bb2)
            draw_axis(pose, camera, rgb, alt_color=False)

        except Exception as e:
            print(e)

        out_path = os.path.join(exp_dir, f'frames_{pose_name}')
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "{:05d}.png".format(i))
        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    out.release()


def generate_videos(exps_dir, data_dir, seqs_dir, pose_name='pem'):
    experiments = os.listdir(exps_dir)
    for exp in experiments:
        if not os.path.isdir(os.path.join(args.dir, exp)):
            continue
        exp_dir = os.path.join(exps_dir, exp)
        seq_dir = os.path.join(seqs_dir, exp)
        print(f"generating video for {exp}")
        generate_video(exp_dir, data_dir, seq_dir, pose_name=pose_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile errors and ratios from selected results folder.")
    # parser.add_argument('--dir', type=str, help='Directory containing the results to compile.',
    #                      default='../Data/demo_pem/output_test_behave')
    parser.add_argument('--dir', type=str, help='Directory containing the results to compile.',
                         default='../Data/demo_pem/output_mem/behave')
                        # default='../Data/demo_pem/output_test_compile')
    parser.add_argument('--data_dir', type=str, help='Directory containing the data.',
                         default='../Data/demo_pem')
    parser.add_argument('--seq_dir', type=str, help='Directory to the sequences',
                         default='/home-local2/chren50.extra.nobkp/behave/test/frames_resized')
    args = parser.parse_args()

    plot_cnos_ratios(args.dir)
    compile_errors(args.dir)
    compile_ratios(args.dir)
    generate_videos(args.dir, args.data_dir, args.seq_dir, pose_name='pem')
    generate_videos(args.dir, args.data_dir, args.seq_dir, pose_name='refiner')
    generate_videos(args.dir, args.data_dir, args.seq_dir, pose_name='fp_only')
