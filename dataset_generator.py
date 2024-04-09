import json
import math
import random

from PIL import Image, ImageDraw


def generate_shape(draw, quadrant):
    color = random.choice(["red", "green", "blue"])
    shape_type = random.choice(["circle", "square", "triangle"])

    size = 8

    if quadrant == 1:
        x = random.randint(size, 31 - size)
        y = random.randint(size, 31 - size)
    elif quadrant == 2:
        x = random.randint(32 + size, 63 - size)
        y = random.randint(size, 31 - size)
    elif quadrant == 3:
        x = random.randint(size, 31 - size)
        y = random.randint(32 + size, 63 - size)
    elif quadrant == 4:
        x = random.randint(32 + size, 63 - size)
        y = random.randint(32 + size, 63 - size)

    if shape_type == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=color)
    elif shape_type == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=color)
    elif shape_type == "triangle":
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        draw.polygon(points, fill=color)

    shape_info = {
        "category_id": ["circle", "square", "triangle"].index(shape_type) + 1,
        "bbox": [x - size, y - size, 2 * size, 2 * size],
        "segmentation": get_segmentation(shape_type, x, y, size),
    }

    return shape_info


def get_segmentation(shape_type, x, y, size, P=30):

    if shape_type == "circle":
        theta = [2 * math.pi * i / P for i in range(P)]
        circle_points = [
            (x + (size + 1) * math.cos(angle), y + (size + 1) * math.sin(angle))
            for angle in theta
        ]

        # Flatten the list of points to create a segmentation mask
        return [coord for point in circle_points for coord in point]

    elif shape_type == "square":
        return [
            x - size - 1,
            y - size - 1,
            x + size + 2,
            y - size - 1,
            x + size + 2,
            y + size + 2,
            x - size - 1,
            y + size + 2,
        ]

    elif shape_type == "triangle":
        return [
            x + 1,
            y - size - 2,

            x - size - 1,
            y + size + 2,

            x + size + 2,
            y + size + 2
        ]


def generate_coco_labels(image_id, width, height, shapes, current_annots):
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
        "categories": [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}, {"id": 3, "name": "triangle"}],
        "annotations": annotations,
        "info": [],
    }

    return coco_labels


def main(N):
    coco_data = {
        'images': [],
        'categories': [],
        'annotations': [],
        'info': []
    }

    for i in range(N):
        img_name = f'data/gen_data/generated_image_{i}.png'
        img = Image.new("RGB", (64, 64), color="white")
        draw = ImageDraw.Draw(img)

        quadrants = [1, 2, 3, 4]
        shapes = []
        for q in quadrants:
            shapes.append(generate_shape(draw, q))

        coco_labels = generate_coco_labels(i, 64, 64, shapes, len(coco_data['annotations']))

        coco_data['images'].append({
            'width': 64,
            'height': 64,
            'id': i,
            'file_name': img_name
        })
        coco_data['categories'] = coco_labels['categories']
        coco_data['annotations'] += coco_labels['annotations']

        img.save(img_name)

    # Save COCO labels in a JSON file
    with open("data/gen_data/coco_labels.json", "w") as json_file:
        json.dump(coco_data, json_file)


if __name__ == "__main__":
    N = 100  # Change N to the desired number of images
    main(N)
