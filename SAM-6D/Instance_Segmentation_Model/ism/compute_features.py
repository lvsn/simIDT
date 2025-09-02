import argparse
import glob
import logging
import os
from PIL import Image
import numpy as np
from hydra import initialize, compose
from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from utils.bbox_utils import CropResizePad
logging.basicConfig(level=logging.INFO)


def compute_features(objects_dir):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='descriptor_only.yaml')

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    logging.info(f"Initalized model: {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    logging.info(f"Moving models to {device} done!")

    objects = os.listdir(objects_dir)
    print(objects)
    for obj in objects:
        logging.info(f"Computing features for object {obj}")
        template_dir = os.path.join(objects_dir, obj, 'templates')
        num_templates = len(glob.glob(f"{template_dir}/rgb/*.png"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb', '{:05d}.png'.format(int(idx))))
            mask = Image.open(os.path.join(template_dir, 'masks', '{:05d}.png'.format(int(idx))))
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        logging.info("Process templates")

        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

        # save_image(templates, f"{template_dir}/templates.png", nrow=7)
        # save_image(masks_cropped, f"{template_dir}/masks.png", nrow=7)

        descriptors = model.descriptor_model.compute_features(templates, token_name="x_norm_clstoken")
        appe_descriptors = model.descriptor_model.compute_masked_patch_feature(templates, masks_cropped[:, 0, :, :])
        torch.save(descriptors, f"{template_dir}/descriptors.pt")
        torch.save(appe_descriptors, f"{template_dir}/appe_descriptors.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", nargs="?", help="Path to root directory of the objects and templates")
    args = parser.parse_args()

    # os.makedirs(f"{args.template_dir}/cnos_results", exist_ok=True)
    objs_dir = os.path.join(args.data_dir, "objects")
    compute_features(objs_dir)