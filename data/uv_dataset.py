import torch
from typing import Optional, List, Dict
import copy
import cv2
from allegroai import DataView, SingleFrame
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
from data.preprocess import apply_preprocess


class UvDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_config: Dict,
            preprocess_config: Dict,
            allegro_frames: Optional[List[SingleFrame]] = None,
            class_list: Optional[List[str]] = None
    ):
        self._dataset_config = dataset_config
        self._preprocess_config = preprocess_config
        self.class_list = class_list

        if allegro_frames is not None:
            self.frames = allegro_frames
        else:
            dv = self._build_data_view()
            self.frames = dv.to_list()

        self.frames = self._expand_annotations()

    def __getitem__(self, index):
        padding_type = self._preprocess_config.get('padding_type', 'sym_edge')
        padding_value = self._preprocess_config.get('padding_value', 25)
        # load image
        frame = self.frames[index]
        image_path = local_source_video_aware(frame)
        image_data = cv2.imread(image_path)
        annotation = frame.annotations

        bbox = annotation[0].get_bounding_box()
        processed_image = apply_preprocess(image_data, bbox, padding_type, padding_value)
        return processed_image, image_path, bbox

    def __len__(self):
        return len(self.frames)

    def _build_data_view(self) -> DataView:
        allegro_dataset_config = self._dataset_config
        mapping_labels_from = allegro_dataset_config.get('mapping_labels_from')
        mapping_labels_to = allegro_dataset_config.get('mapping_labels_to')

        dataview = DataView(auto_connect_with_task=False)
        for query in allegro_dataset_config.get('queries'):
            dataview.add_query(
                dataset_name=query.get('dataset_name'),
                version_name=query.get('version_name'),
                frame_query=query.get('frame_query'),
                roi_query=query.get('roi_query')
            )

            if len(mapping_labels_from) and len(mapping_labels_to):
                assert len(mapping_labels_from) == len(mapping_labels_to)
                for src_label, dst_label in zip(mapping_labels_from, mapping_labels_to):
                    dataview.add_mapping_rule(
                        dataset_name=query.get('dataset_name'),
                        version_name=query.get('version_name'),
                        from_labels=[src_label],
                        to_label=dst_label
                    )

        return dataview

    def _expand_annotations(self):
        expanded_data = []
        for index, frame in enumerate(self.frames):
            annotations = frame.annotations
            if self.class_list is not None:
                annotations = [a for a in annotations if a.labels[0] in self.class_list]
            for ann in annotations:
                data = copy.deepcopy(self.frames[index])
                data.annotations = [ann]
                expanded_data.append(data)
        return expanded_data
