import importlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_module_from_string(module_path):
    module_breakdown = module_path.split('.')
    loss_fn = module_breakdown.pop()
    loss_fn_module = importlib.import_module('.'.join(module_breakdown))
    return getattr(loss_fn_module, loss_fn)


def get_label_map(label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f).get('label_map')

    if not label_map:
        raise ValueError('Empty or missing label map')

    rev = {}
    for k, v in label_map.items():
        rev[v] = k

    return label_map, rev


def generate_masks_binary(coco, save_dir):
    save_dir = Path(save_dir)
    cat_ids = coco.getCatIds()

    for img in tqdm(coco.imgs.values()):
        # get all annotations for this image
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        # create a mask using label encoding (each pixel is a value representing the class it belongs to)
        mask = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        for a in anns:
            m = coco.annToMask(a)
            mask[m > 0] = 255

        filename = Path(img['file_name']).stem + '.png'
        Image.fromarray(mask).save(save_dir.joinpath(filename))


def generate_masks_multiclass(coco, save_dir, random_seed=42):
    random.seed(random_seed)
    save_dir = Path(save_dir)
    cat_ids = coco.getCatIds()

    class_colors = []
    for _ in cat_ids:

        rand_color = random.choices(range(256), k=3)
        class_colors.append(rand_color)

    for img in tqdm(coco.imgs.values()):
        try:
            # get all annotations for this image
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)

            # create a mask using label encoding (each pixel is a value representing the class it belongs to)
            mask = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
            for a in anns:
                m = coco.annToMask(a)
                color = class_colors[a['category_id']]
                mask[m > 0] = color

            filename = Path(img['file_name']).stem + '.png'
            Image.fromarray(mask).save(save_dir.joinpath(filename))
        except Exception as e:
            print(f'Failed to create mask for {img["file_name"]}: {e}')
