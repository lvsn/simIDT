import trimesh
import numpy as np
import os
import argparse
import torch
import random
import gorilla
from pem.model import pose_estimation_model
from pem.run_inference_batch import get_templates

from evaluate_sequence_tracker_behave import init_pem

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, default="0", help="path to pretrain model")
parser.add_argument("--model", type=str, default="pose_estimation_model", help="path to model file")
parser.add_argument("--config", type=str, default="Pose_Estimation_Model/pem/config/base.yaml", help="path to config file, different config.yaml use different config")
parser.add_argument("--iter", type=int, default=600000, help="epoch num. for testing")
parser.add_argument("--exp_id", type=int, default=0, help="")

args = parser.parse_args()
pem_cfg = init_pem(args)

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

all_pts = []
true_all_tem_pts = []
true_all_tem_feats = []
objects_dir = './Data/demo_pem/objects_behave'
objects = os.listdir(objects_dir)
object_names = [o for o in objects if os.path.isdir(f"{objects_dir}/{o}")]

N_CAD = 1024
N_TEM = 2048

for object_name in object_names:
    # CAD pts
    print(f'Processing {object_name}')
    cad_path_real = [f"{objects_dir}/{object_name}/{o}" for o in os.listdir(f"{objects_dir}/{object_name}") if o.endswith('.obj')][0]
    mesh : trimesh.Trimesh = trimesh.load(cad_path_real, force='mesh')
    print(mesh.scale)
    model_points = mesh.sample(N_CAD).astype(np.float32)
    if mesh.scale > 10:
        print('apply scale')
        model_points = model_points / 1000.0
    all_pts.append(model_points)
    np.save(os.path.join(objects_dir, object_name, 'templates', 'all_pts.npy'), model_points)

    # TMP pts
    tem_path = os.path.join(objects_dir, object_name, 'templates')
    all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, pem_cfg.test_dataset)
    with torch.no_grad():
        all_tem_pts, all_tem_feat = pem_model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)
    true_all_tem_pts.append(all_tem_pts)
    true_all_tem_feats.append(all_tem_feat)
    torch.save(all_tem_pts, os.path.join(objects_dir, object_name, 'templates', 'all_tem_pts.pth'))
    torch.save(all_tem_feat, os.path.join(objects_dir, object_name, 'templates', 'all_tem_feats.pth'))

all_pts = np.stack(all_pts, axis=0)
print(all_pts.shape)
np.save(os.path.join(objects_dir, 'all_pts.npy'), all_pts)
true_all_tem_pts = torch.stack(true_all_tem_pts, axis=0)
true_all_tem_pts.squeeze_(1)
print(true_all_tem_pts.shape)
torch.save(true_all_tem_pts, os.path.join(objects_dir, 'all_tem_pts.pth'))
true_all_tem_feats = torch.stack(true_all_tem_feats, axis=0)
true_all_tem_feats.squeeze_(1)
print(true_all_tem_feats.shape)
torch.save(true_all_tem_feats, os.path.join(objects_dir, 'all_tem_feats.pth'))