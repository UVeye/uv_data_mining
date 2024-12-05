from __future__ import annotations
from typing import Dict, List
import enum
import os
import json
import cv2
import matplotlib.pyplot as plt
import argparse


def select_top_10(data):
    results = {}
    for case, items in data.items():
        # Sort the items by 'embed_dist' in ascending order
        sorted_items = sorted(items, key=lambda x: x["embed_dist"])
        # Select the top 10
        top_10 = sorted_items[:10]
        # Combine "image_path" and "bbox" for each item
        results[case] = [
            {"image_path": item["image_path"], "bbox": item["bbox"]}
            for item in top_10
        ]
    return results


def showcase_query_samples(queries_dir: str, queries_output_dict: Dict, output_path: str):
    samples = select_top_10(queries_output_dict)

    queries = os.listdir(queries_dir)

    for idx, q in enumerate(queries):
        query_path = os.path.join(queries_dir, q)
        query_img = (cv2.imread(query_path))

        case_samples = samples[f'case_{idx}']
        sample_images = []
        for sample in case_samples:
            image = process_image_to_crop(sample)
            sample_images.append(image)

        case_output_path = os.path.join(output_path, f"showcase_results_{idx}.png")
        create_showcase_figure(query_img, f'case_{idx}', sample_images, case_output_path)


def process_image_to_crop(image_box_dict: Dict):
    image_path = image_box_dict.get('image_path')
    image = cv2.imread(image_path)
    bbox = image_box_dict.get('bbox')
    cropped_image = image[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])].copy()
    return cropped_image


def create_showcase_figure(query_img, query_name: str, retrieved_images: List, output_path: str):
    num_images = len(retrieved_images)
    #retrieved_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in retrieved_images]
    #query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    num_cols = 5
    num_rows = (num_images // num_cols) + (1 if num_images % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows + 3, num_cols, figsize=(num_cols * 3, (num_rows + 3) * 3))
    axes = axes.flatten()

    for i in range(num_cols):  # Spanning all columns
        if i == 0:  # Add text in the first column
            axes[i].text(0.5, 0.5, f'Query: {query_name}', fontsize=22, ha='center', va='center', fontweight='bold')
        axes[i].axis('off')  # Hide the axis

    for i in range(num_cols):
        if i == 0:  # Add the source image in the first column
            axes[num_cols + i].imshow(query_img)
        axes[num_cols + i].axis('off')  # Hide the axis

    for i in range(num_cols):  # Spanning all columns
        if i == 0:  # Add text in the first column
            axes[2 * num_cols + i].text(0.5, 0.5, 'Retrieved images (samples)', fontsize=22, ha='center', va='center', fontweight='bold')
        axes[2 * num_cols + i].axis('off')  # Hide the axis

    for i, img in enumerate(retrieved_images):
        row_start = 3 * num_cols  # Retrieved images start after the first 3 rows
        ax = axes[row_start + i]
        ax.imshow(img)
        ax.axis('off')  # Hide axis ticks and labels

    for j in range(row_start + len(retrieved_images), len(axes)):
        axes[j].axis('off')

    # Adjust layout to ensure titles and images fit well
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.savefig(output_path, bbox_inches='tight', dpi=300)


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--queries_dir", type=str, required=True, help="image to visualize")
    parser.add_argument("--queries_output_dict", type=str, required=True, help="json dict for the mapping of indices to images and bbox")
    parser.add_argument("--output_path", type=str, required=True, help="folder to save the results to")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.queries_output_dict, 'r') as json_file:
        queries_output_dict = json.load(json_file)

    showcase_query_samples(args.queries_dir, queries_output_dict, args.output_path)


if __name__ == "__main__":
    main()
