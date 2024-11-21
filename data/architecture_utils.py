
def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of (processed_image, image_path, bbox).
    """
    processed_images = []
    image_paths = []
    bboxes = []
    for item in batch:
        processed_images.append(item[0])  # processed_image
        image_paths.append(item[1])      # image_path
        bboxes.append(item[2])           # bbox (list of 4 values)

    return processed_images, image_paths, bboxes