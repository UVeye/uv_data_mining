from algo_flows.general_utils.base_utils import create_padding_bbox


def apply_preprocess(image, bbox, padding_type, padding_value):
    padded_bbox = create_padding_bbox(
        box_ltwh=[int(v) for v in bbox],
        pad_type=padding_type,
        pad_value=padding_value,
        source_image_shape_hw=image.shape,
        allow_exceed_image=False
    )
    return image[padded_bbox[1]:padded_bbox[1] + padded_bbox[3], padded_bbox[0]:padded_bbox[0] + padded_bbox[2]]