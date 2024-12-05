import os
import yaml
import json
import argparse
import faiss
import torch
import numpy as np
from PIL import Image
import cv2
from concurrent import futures
from tqdm import tqdm
from ultralytics import YOLO
from typing import Optional, List

from data.uv_dataset import UvDataset
from data.architecture_utils import custom_collate_fn


def create_index_vector(cfg, weights: str, output_dir: str, n_workers: int, batch_size: int, class_list: Optional[List[str]] = None):
    data_config = cfg.get('allegro_dataset')
    preprocess_config = cfg.get('preprocessor_config')
    dataset = build_dataset(data_config, preprocess_config, class_list)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=n_workers,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    model = YOLO(weights)

    output_path = os.path.join(output_dir, 'vector_part_damages.json')
    data_dict = {}
    for idx, (images, images_paths, bboxes) in enumerate(tqdm(data_loader)):
        print(f"Processing batch {idx}: {batch_size} images")
        index = generate_embeddings(images, model)
        output_idx_path = os.path.join(output_dir, f'vector_part_damages{idx}.index')
        # Save the index
        faiss.write_index(index, output_idx_path)
        print(f"Index created and saved to {output_dir}")

        for _, path, bbox in zip(images, images_paths, bboxes):
            if idx not in data_dict:
                data_dict[idx] = {'image_paths_bboxes': []}
            data_dict[idx]['image_paths_bboxes'].append({'image_path': path, 'bbox': bbox})

        # Sort the image_paths_bboxes list by image_path (or any other criterion you want)
        for idx, data in data_dict.items():
            data['image_paths_bboxes'] = sorted(data['image_paths_bboxes'], key=lambda x: x['image_path'])

        # Now write to JSON, ensuring all values are serializable
        with open(output_path, 'w') as json_file:
           json.dump([{'index_path': values['index_path'],
                   'image_paths_bboxes': values['image_paths_bboxes']}
                  for index, values in data_dict.items()],
                 json_file, indent=4)
        print(f"Updated JSON file saved to {output_path}")


def build_dataset(data_config, preprocess_config, class_list: Optional[List[str]] = None, cache_path: Optional[str] = None):
    return UvDataset(dataset_config=data_config, preprocess_config=preprocess_config, class_list=class_list)


def generate_embeddings(images, model):
    index = faiss.IndexFlatL2(1280)

    def process_frame(image):
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                results = model.predict(img, embed=[18, 21, 15])  # Adjust layers as needed
            add_vector_to_index(results[0], index)
            if np.mod(index.ntotal, 100) == 0:
                print(index.ntotal)
        except Exception as e:
            print("Error:", str(e))

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_list = [executor.submit(process_frame, image) for image in tqdm(images, desc="Parsing frames")]
    return index


def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    vector = np.expand_dims(vector, axis=0)
    faiss.normalize_L2(vector)
    index.add(vector)


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--data_config_path", type=str, required=False, help="yaml config for the data",
                        default='/home/barrm/workspace/uv_data_mining/configs/data_conf.yaml')
    parser.add_argument("--weights", type=str, required=True, help="weights for YOLO model")
    parser.add_argument("--output_dir", type=str, required=True, help="folder to save the results to")
    parser.add_argument("--n_workers", type=int, required=False, default=4)
    parser.add_argument("--batch_size", type=int, required=False, default=1000)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.data_config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    class_list = config.get('class_list', None)
    create_index_vector(config, args.weights, args.output_dir, args.n_workers, args.batch_size, class_list)


if __name__ == "__main__":
    main()
