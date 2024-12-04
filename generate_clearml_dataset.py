import os
import json
import yaml
import argparse
from tqdm import tqdm
from allegroai import DataView, DatasetVersion
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware


def get_common_path_up_to_parent(path, parent_name):
    parts = path.split(os.sep)
    index = parts.index(parent_name)
    return os.sep.join(parts[index:])


def create_dataset_w_imgs_list(data_dict, data_config):
    match_images_paths = []
    for case, items in data_dict.items():
        match_images_paths.extend(get_common_path_up_to_parent(item['image_path'], 's3') for item in items)

    dataset_name = data_config.get('dataset_name')
    version_name = data_config.get('version_name')
    dv = DataView()
    dv.add_query(
        dataset_name=dataset_name,
        version_name=version_name
    )
    infer_frames = dv.to_list()
    save_frames = []
    for frame in tqdm(infer_frames):
        frame_path = local_source_video_aware(frame)
        if get_common_path_up_to_parent(frame_path, 's3') in match_images_paths:
            save_frames.append(frame)

    new_version_name = f"data_mining_v0__{version_name.split('__')[1]}"
    new_version = DatasetVersion.create_version(
        dataset_name=dataset_name,
        version_name=new_version_name)

    new_version.add_frames(save_frames)


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--retrieved_images_path", type=str, required=True, help="json dict for the retrieved matches")
    parser.add_argument("--config_path", type=str, required=False, help="yaml config for the data")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    with open(args.retrieved_images_path, 'r') as json_file:
        data_dict = json.load(json_file)

    create_dataset_w_imgs_list(data_dict, config)


if __name__ == "__main__":
    main()
