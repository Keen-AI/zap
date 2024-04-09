import json
import os

import numpy as np
import pandas as pd


def csv_to_coco(data_dir):
    """Convert csv data format of column_names =['filename','label','width', 'height','xmin','ymin','xmax','ymax'] into coco json

    Args:
        csv_path (str): path to csv
        save_json_path (str): path to json

    """

    csv_path = os.path.join(data_dir, "dataset.csv")
    save_json_path = os.path.join(data_dir, "dataset.json")

    data = pd.read_csv(csv_path)

    images = []
    categories = []
    annotations = []

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['categoryid'] = pd.Categorical(data['label'], ordered=True).codes
    data['categoryid'] = data['categoryid'] + 1
    data['annid'] = data.index

    def image(row):
        image = {}
        image["height"] = row.height
        image["width"] = row.width
        image["id"] = row.fileid
        image["file_name"] = row.filename
        return image

    def category(row):
        category = {}
        category["supercategory"] = 'None'
        category["id"] = row.categoryid
        category["name"] = row.label.lower()
        return category

    def annotation(row):
        annotation = {}
        area = (row.xmax - row.xmin) * (row.ymax - row.ymin)
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = row.fileid

        annotation["bbox"] = [row.xmin, row.ymin, row.xmax - row.xmin, row.ymax - row.ymin]

        annotation["category_id"] = row.categoryid
        annotation["id"] = row.annid
        return annotation

    for row in data.itertuples():
        annotations.append(annotation(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
    for row in catdf.itertuples():
        categories.append(category(row))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    json.dump(data_coco, open(save_json_path, "w"), indent=4)
