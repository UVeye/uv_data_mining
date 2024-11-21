import os
import io
import json
import random
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse

from data.preprocess import apply_preprocess


def showcase_query_samples(query_path: str, queries_output_dict: Dict, output_path: str):
    query_img = cv2.imread(query_path)
    query_name = os.path.splitext(os.path.basename(query_path))[0]
    query_output = queries_output_dict.get(query_name)
    random_indices = random.sample(range(len(query_output)), min(10, len(query_output)))

    sample_images = []
    for idx in random_indices:
        result = query_output[idx]
        image = process_image_to_crop(result)
        sample_images.append(image)

    create_showcase_figure(query_img, query_name, sample_images, output_path)


def process_image_to_crop(image_box_dict: Dict):
    image_path = image_box_dict.get('image_path')
    image = cv2.imread(image_path)
    bbox = image_box_dict.get('bbox')
    cropped_image = apply_preprocess(image, bbox,  'sym_edge', 0)
    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    return cropped_image_pil


def create_showcase_figure(query_img, query_name: str, retrieved_images: List, output_path: str):
    num_images = len(retrieved_images)

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

    plt.savefig(os.path.join(output_path, "showcase_results.png"), bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--query_path", type=str, required=True, help="image to visualize")
    parser.add_argument("--queries_output_dict", type=str, required=True, help="json dict for the mapping of indices to images and bbox")
    parser.add_argument("--output_path", type=str, required=True, help="folder to save the results to")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.queries_output_dict, 'r') as json_file:
        queries_output_dict = json.load(json_file)

    showcase_query_samples(args.query_path, queries_output_dict, args.output_path)


if __name__ == "__main__":
    main()
