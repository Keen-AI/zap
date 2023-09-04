import json

def get_label_map(label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f).get('label_map')

    if not label_map:
        raise ValueError('Empty or missing label map')

    rev = {}
    for k, v in label_map.items():
        rev[v] = k

    return label_map, rev