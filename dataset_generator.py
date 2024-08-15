import csv
import json
import math
import random

from PIL import Image, ImageDraw

MODE = 'classification'  # or coco
NUMBER_OF_IMAGES = 100
IMAGE_SIZE = 244
SHAPE_SIZE = IMAGE_SIZE // 10
SAVE_DIR = 'data/gen_data'


def _generate_shape(draw, quadrant):
    color = random.choice(["red", "green", "blue"])
    shape_type = random.choice(["circle", "square", "triangle"])

    if quadrant == 1:
        x = random.randint(SHAPE_SIZE, IMAGE_SIZE // 2 - SHAPE_SIZE)
        y = random.randint(SHAPE_SIZE, IMAGE_SIZE // 2 - SHAPE_SIZE)
    elif quadrant == 2:
        x = random.randint(IMAGE_SIZE // 2 + SHAPE_SIZE, IMAGE_SIZE - SHAPE_SIZE)
        y = random.randint(SHAPE_SIZE, IMAGE_SIZE // 2 - SHAPE_SIZE)
    elif quadrant == 3:
        x = random.randint(SHAPE_SIZE, IMAGE_SIZE // 2 - SHAPE_SIZE)
        y = random.randint(IMAGE_SIZE // 2 + SHAPE_SIZE, IMAGE_SIZE - SHAPE_SIZE)
    elif quadrant == 4:
        x = random.randint(IMAGE_SIZE // 2 + SHAPE_SIZE, IMAGE_SIZE - SHAPE_SIZE)
        y = random.randint(IMAGE_SIZE // 2 + SHAPE_SIZE, IMAGE_SIZE - SHAPE_SIZE)

    if shape_type == "circle":
        draw.ellipse((x - SHAPE_SIZE, y - SHAPE_SIZE, x + SHAPE_SIZE, y + SHAPE_SIZE), fill=color)
    elif shape_type == "square":
        draw.rectangle((x - SHAPE_SIZE, y - SHAPE_SIZE, x + SHAPE_SIZE, y + SHAPE_SIZE), fill=color)
    elif shape_type == "triangle":
        points = [(x, y - SHAPE_SIZE), (x - SHAPE_SIZE, y + SHAPE_SIZE), (x + SHAPE_SIZE, y + SHAPE_SIZE)]
        draw.polygon(points, fill=color)

    shape_info = {
        "category_id": ["circle", "square", "triangle"].index(shape_type),
        "bbox": [x - SHAPE_SIZE, y - SHAPE_SIZE, 2 * SHAPE_SIZE, 2 * SHAPE_SIZE],
        "segmentation": _get_segmentation(shape_type, x, y, SHAPE_SIZE),
    }

    return shape_info, shape_type


def _get_segmentation(shape_type, x, y, SHAPE_SIZE, P=30):

    if shape_type == "circle":
        theta = [2 * math.pi * i / P for i in range(P)]
        circle_points = [
            (x + (SHAPE_SIZE + 1) * math.cos(angle), y + (SHAPE_SIZE + 1) * math.sin(angle))
            for angle in theta
        ]

        # Flatten the list of points to create a segmentation mask
        return [coord for point in circle_points for coord in point]

    elif shape_type == "square":
        return [
            x - SHAPE_SIZE - 1,
            y - SHAPE_SIZE - 1,
            x + SHAPE_SIZE + 2,
            y - SHAPE_SIZE - 1,
            x + SHAPE_SIZE + 2,
            y + SHAPE_SIZE + 2,
            x - SHAPE_SIZE - 1,
            y + SHAPE_SIZE + 2,
        ]

    elif shape_type == "triangle":
        return [
            x + 1,
            y - SHAPE_SIZE - 2,

            x - SHAPE_SIZE - 1,
            y + SHAPE_SIZE + 2,

            x + SHAPE_SIZE + 2,
            y + SHAPE_SIZE + 2
        ]


def _generate_coco_labels(image_id, width, height, shapes, current_annots):
    annotations = []

    for idx, shape in enumerate(shapes):
        annotation = {
            "id": current_annots + idx,
            "image_id": image_id,
            "category_id": shape["category_id"],
            "bbox": shape["bbox"],
            "area": shape["bbox"][2] * shape["bbox"][3],
            "iscrowd": 0,
            "segmentation": [shape["segmentation"]],
        }

        annotations.append(annotation)

    coco_labels = {
        "categories": [{"id": 0, "name": "circle"}, {"id": 1, "name": "square"}, {"id": 2, "name": "triangle"}],
        "annotations": annotations,
        "info": [],
    }

    return coco_labels


def generate_coco_data():
    coco_data = {
        'images': [],
        'categories': [],
        'annotations': [],
        'info': []
    }

    for i in range(NUMBER_OF_IMAGES):
        img_name = f'generated_image_{i}.png'
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")
        draw = ImageDraw.Draw(img)

        quadrants = [1, 2, 3, 4]
        shapes = []
        for q in quadrants:
            shapes.append(_generate_shape(draw, q)[0])

        coco_labels = _generate_coco_labels(i, IMAGE_SIZE, IMAGE_SIZE, shapes, len(coco_data['annotations']))

        coco_data['images'].append({
            'width': IMAGE_SIZE,
            'height': IMAGE_SIZE,
            'id': i,
            'file_name': img_name
        })
        coco_data['categories'] = coco_labels['categories']
        coco_data['annotations'] += coco_labels['annotations']

        img.save(f'{SAVE_DIR}images/{img_name}')

    # Save COCO labels in a JSON file
    with open(f"{SAVE_DIR}labels.json", "w") as json_file:
        json.dump(coco_data, json_file)


def generate_classification_data():
    classifier_data = []
    for i in range(NUMBER_OF_IMAGES):
        img_name = f'generated_image_{i}.png'
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")
        draw = ImageDraw.Draw(img)

        _, shape = _generate_shape(draw, 1)

        classifier_data.append({'filename': img_name, 'label': shape})

        img.save(f'{SAVE_DIR}/images/{img_name}')

    with open(f"{SAVE_DIR}/labels.csv", "w", newline='') as csv_file:
        column_names = ['filename', 'label']
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        csv_writer.writerows(classifier_data)


if __name__ == "__main__":
    if MODE == 'classification':
        generate_classification_data()
    elif MODE == 'coco':
        generate_coco_data()
