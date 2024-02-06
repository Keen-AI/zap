from transformers import DetaImageProcessor


processor = DetaImageProcessor.from_pretrained(
    "jozhang97/deta-resnet-50",
    size={
        "shortest_edge": 400,
        "longest_edge": 666})


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch
