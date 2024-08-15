"""
A set of utilities to help clean COCO JSON data

"""

import json
import os

from pycocotools.coco import COCO
from tqdm import tqdm


# TODO: either use plain json/dict or pycocotools, not a mix of both
class COCOUtils():
    def __init__(self, coco_path) -> None:
        self.coco_path = coco_path
        with open(coco_path, 'r') as f:
            self.coco_data = json.load(f)
            self.original_coco_data = self.coco_data.copy()

        self.coco = COCO(coco_path)

    def write_json(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.coco_data, f)

    def get_class_distribution(self):
        cat_ids = self.coco.getCatIds()
        cat_names = [self.coco.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

        label_counts = {cat_name: 0 for cat_name in cat_names}

        ann_ids = self.coco.getAnnIds()
        annotations = self.coco.loadAnns(ann_ids)

        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = self.coco.loadCats(cat_id)[0]['name']
            label_counts[cat_name] += 1

        total_labels = sum(label_counts.values())

        print(f'TOTAL LABELS: {total_labels}')

        for cat_name, count in label_counts.items():
            percentage = (count / total_labels) * 100
            print(f"{cat_name}: {count} labels ({percentage:.2f}% of total)")

        return label_counts

    def filter_classes(self, categories):
        # create mapping from category names to category IDs
        categories_map = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        filtered_categories = [cat for cat in self.coco_data['categories'] if cat['name'] in categories]
        reindexed_categories = []
        reindexed_map_old_to_new = {}
        for i, cat_entry in enumerate(filtered_categories):
            reindexed_categories.append({'id': i, 'name': cat_entry['name']})
            reindexed_map_old_to_new[cat_entry['id']] = i

        filtered_images = []
        filtered_annotations = []
        for img in tqdm(self.coco_data['images']):
            for ann in self.coco_data['annotations']:
                cat_name = categories_map[ann['category_id']]
                ann_copy = ann.copy()
                if ann['image_id'] == img['id'] and cat_name in categories:
                    img['include'] = True
                    filtered_images.append(img)
                    ann_copy['category_id'] = reindexed_map_old_to_new[ann_copy['category_id']]
                    filtered_annotations.append(ann_copy)

        # remove duped images
        unique_filtered_images = []
        for img in filtered_images:
            if img.get('include', False):
                img.pop('include', None)
                unique_filtered_images.append(img)

        # create new COCO JSON structure
        filtered_coco_data = {
            "images": unique_filtered_images,
            "categories": reindexed_categories,
            "annotations": filtered_annotations
        }

        self.coco_data = filtered_coco_data
        return self.coco_data

    def download_images(self, prefix, save_path, batch_size=100, dry_run=False):
        images = self.coco_data['images']

        # ensure the prefix ends in a /
        if prefix[-1] != '/':
            prefix += '/'

        # yes, we use the -n flag in gsutil to skip existing images
        # but this reduces the overall work we're doing
        existing_images = []
        for (_, _, filenames) in os.walk(save_path):
            existing_images.extend(filenames)
            break

        os.makedirs('download_logs', exist_ok=True)

        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i + batch_size]
            cmd = f"gsutil -m cp -n -L download_logs/batch_{i/batch_size}.log -r "
            for img in batch:
                if img['file_name'] in existing_images:
                    continue
                filename = f'"{prefix}{img["file_name"]}"'
                cmd += filename + ' '

            # if we haven't added any files to the command, don't execute it
            if prefix not in cmd:
                continue

            cmd += save_path

            if dry_run:
                print('DRY RUN - THE FOLLOWING COMMAND WAS NOT EXECUTED')
                print(cmd)
            else:
                os.system(cmd)

    def stratify_data(self):
        pass

    def add_null_category(self):
        # increase each category index by 1 and update annotation records
        for category in self.coco_data['categories']:
            category['id'] += 1

        for annotation in self.coco_data['annotations']:
            annotation['category_id'] += 1

        # create a new category record at index 0
        nothing_category = {
            "id": 0,
            "name": "NOTHING",
        }
        self.coco_data['categories'].insert(0, nothing_category)
        return self.coco_data

    def remove_images(self, images):
        filtered_images = []
        filtered_annotations = []

        for img in tqdm(self.coco_data['images']):
            if img['file_name'] not in images:
                for ann in self.coco_data['annotations']:
                    if ann['image_id'] == img['id']:
                        img['include'] = True
                        filtered_images.append(img)
                        filtered_annotations.append(ann)

        unique_filtered_images = []
        for img in filtered_images:
            if img.get('include', False):
                img.pop('include', None)
                unique_filtered_images.append(img)

        filtered_coco_data = {
            "images": unique_filtered_images,
            "categories": self.coco_data['categories'],
            "annotations": filtered_annotations
        }

        self.coco_data = filtered_coco_data
        return self.coco_data

    def remove_labeless_images(self):
        img_ids = self.coco.getImgIds()
        ann_img_ids = set([ann['image_id'] for ann in self.coco.dataset['annotations']])

        images_with_no_annotations = [img_id for img_id in img_ids if img_id not in ann_img_ids]

        return self.remove_images(images_with_no_annotations)

    def remove_degenerate_bboxes(self):
        images_to_remove = []
        for image in self.coco_data['images']:
            image_id = image['id']
            annotations_to_remove = []
            for annotation in self.coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    bbox = annotation['bbox']
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        annotations_to_remove.append(annotation['id'])
            if annotations_to_remove:
                images_to_remove.append(image_id)
                for annotation_id in annotations_to_remove:
                    self.coco_data['annotations'] = [
                        annotation for annotation in self.coco_data['annotations'] if annotation['id'] != annotation_id]

        self.coco_data['images'] = [image for image in self.coco_data['images'] if image['id'] not in images_to_remove]

        print(f"Removed {len(images_to_remove)} images with degenerate bounding boxes and their annotations.")

        return self.coco_data
