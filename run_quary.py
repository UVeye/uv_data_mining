import os
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import faiss
from ultralytics import YOLO
from typing import Dict, List


def run_query(weights: str, input_path: str, data_dict: Dict, output_dir: str):
    assert os.path.isdir(input_path), f"The path {input_path} is not a valid directory."

    model = YOLO(weights)

    query_results = {}
    query_embed_dist = {}
    queries = os.listdir(input_path)
    for q in queries:
        query_path = os.path.join(input_path, q)
        query_name = os.path.splitext(q)[0]
        retrieved_images, retrieved_distances = [], []
        for entry in data_dict:
            index_path = entry['index_path']
            image_paths_bboxes = entry['image_paths_bboxes']

            results = similarity_search(index_path, image_paths_bboxes, model, query_path, 10)
            retrieved_images += results[0]
            retrieved_distances.append(results[1])

        retrieved_distances = np.concatenate(retrieved_distances)
        query_embed_dist[query_name] = retrieved_distances
        query_results[query_name] = retrieved_images

    plot_embed_dist_hist(query_embed_dist, output_dir)
    dict_output_path = os.path.join(output_dir, 'retrieved_images.json')
    with open(dict_output_path, 'w') as json_file:
        json.dump(query_results, json_file, indent=4)
    print(f"Query results saved to {dict_output_path}")


def retrieve_similar_images(query, model, index, image_paths_bboxes, top_k=3):
    assert query.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')), \
         f"File '{query}' does not have a valid extension. Accepted extensions are: .png, .jpg, .jpeg, .tiff, .bmp, .gif."
    query = Image.open(query)

    with torch.no_grad():
        query_features = model.predict(query, embed=[18, 21, 15])  # Adjust layers as needed

    vector = query_features[0].detach().cpu().numpy()
    vector = np.float32(vector)
    vector = np.expand_dims(vector, axis=0)
    faiss.normalize_L2(vector)

    distances, indices = index.search(vector, top_k)
    #filtered_indices, filtered_distances = filter_by_distance(distances, indices)
    retrieved_images = [image_paths_bboxes[idx] for idx in indices[0] if 0 <= idx]

    n_images = len(retrieved_images)
    for i, item in enumerate(retrieved_images):
        item['embed_dist'] = round(float(distances[0][i]), 4)

    return retrieved_images, distances[0][0:n_images]


def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index


def similarity_search(index_path, image_paths_bboxes, model, query, top_k):
    index = load_faiss_index(index_path)
    retrieved_images, distances = retrieve_similar_images(query, model, index, image_paths_bboxes, top_k=top_k)

    return retrieved_images, distances


def filter_by_distance(distances, indices, ratio=0.5):
    # Filters the indices and distances based on a ratio of the closest distance.
    closest_distance = distances[0][0]  # The smallest distance (closest neighbor)
    filtered_indices = [indices[i] for i, dist in enumerate(distances[0]) if dist <= closest_distance * ratio]
    filtered_distances = [dist for dist in distances[0] if dist <= closest_distance * ratio]
    return filtered_indices, filtered_distances


def plot_embed_dist_hist(distances_dict: Dict, output_dir: str):
    for query_name in distances_dict.keys():
        distances = distances_dict[query_name]
        plt.figure(figsize=(8, 6))
        plt.hist(distances, bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f'Histogram of query {query_name} embedding distances')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlim(left=0)

        plt.savefig(os.path.join(output_dir, f"quary_{query_name}_embedding_distances.png"), bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--data_json_path", type=str, required=True, help="json dict for the mapping of indices to images and bbox")
    parser.add_argument("--weights", type=str, required=True, help="weights for YOLO model")
    parser.add_argument("--queries", type=str, required=True, help="image or folder for images (crops) to test")
    parser.add_argument("--output_dir", type=str, required=True, help="folder to save the results to")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.data_json_path, 'r') as json_file:
        data_dict = json.load(json_file)

    run_query(args.weights, args.queries, data_dict, args.output_dir)


if __name__ == "__main__":
    main()
