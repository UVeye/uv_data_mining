from __future__ import annotations
from typing import Sequence
import numpy as np
from typing import List, Tuple, Optional
import enum


class PadType(enum.Enum):
    """ Padding types:
    sym_edge:   add constant padding for all 4 sides.
    sym_center: output is square, centered on the input center, sized as the padding size.
                If smaller than input image, output will be cropped.
    sym_square: pad the smaller edge of the box to be the same size as the large part.
                Resulting image is centered in the smaller axis. If pad value is provided,
                then afterwards, sym-edge also happens with the value.
    """

    sym_edge = "sym_edge"
    sym_center = "sym_center"
    sym_square = "sym_square"
    box = "box"


def create_padding_bbox(
        box_ltwh: Sequence,
        pad_type: Optional[PadType],
        pad_value: float | List[float],
        source_image_shape_hw: np.ndarray | Tuple[int, int],
        allow_exceed_image: bool = True
) -> np.ndarray:
    """
    sym-edge: add constant padding for all 4 sides.
    sym-center: output is square, centered on the input center, sized as the padding size.
    If smaller than input image, output will be cropped.
    sym-square: pad the smaller edge of the box to be the same size as the large part.
     Resulting image is centered in the smaller axis. If pad value is provided, then afterwards, sym-edge also happens
     with the value.

    :param box_ltwh:
    :param pad_type:
    :param pad_value:
    :param source_image_shape_hw: source_image_shape_wh
    :param allow_exceed_image: if True, box can have negative values and very large values.
    :return:
    """

    if pad_type is None:
        return np.array(box_ltwh)

    left, top, width, height = box_ltwh
    # right = left + width
    # bottom = top + height

    padded_box_ltwh = np.array(box_ltwh)
    if pad_type == PadType.sym_edge:
        padded_box_ltwh[0] -= pad_value
        padded_box_ltwh[1] -= pad_value
        padded_box_ltwh[2] += 2 * pad_value
        padded_box_ltwh[3] += 2 * pad_value

    elif pad_type == PadType.sym_center:
        center_x, center_y = top + (height / 2), left + (width / 2)
        padded_box_ltwh[0] = int(center_y - pad_value)
        padded_box_ltwh[1] = int(center_x - pad_value)
        padded_box_ltwh[2] = int(2 * pad_value)
        padded_box_ltwh[3] = int(2 * pad_value)

    elif pad_type == PadType.sym_square:
        if padded_box_ltwh[2] > padded_box_ltwh[3]:
            diff = padded_box_ltwh[2] - padded_box_ltwh[3]
            half = diff / 2
            padded_box_ltwh[1] -= half
            padded_box_ltwh[3] += diff
        elif width < height:
            diff = padded_box_ltwh[3] - padded_box_ltwh[2]
            half = diff / 2
            padded_box_ltwh[0] -= half
            padded_box_ltwh[2] += diff
        padded_box_ltwh[0] -= pad_value
        padded_box_ltwh[1] -= pad_value
        padded_box_ltwh[2] += 2 * pad_value
        padded_box_ltwh[3] += 2 * pad_value

    elif pad_type == PadType.box:                      # up, down, left, right in px
        padded_box_ltwh[0] -= pad_value[2]                  # left
        padded_box_ltwh[1] -= pad_value[0]                  # up
        padded_box_ltwh[2] += pad_value[2] + pad_value[3]   # width
        padded_box_ltwh[3] += pad_value[0] + pad_value[1]   # height

    if allow_exceed_image:
        padded_box_ltwh = np.array([int(b) for b in padded_box_ltwh])
    else:
        world_height, world_width = source_image_shape_hw[:2]
        padded_box_ltwh = [int(b) if b > 0 else 0 for b in padded_box_ltwh]
        if padded_box_ltwh[0] + padded_box_ltwh[2] >= world_width:
            padded_box_ltwh[2] = world_width - padded_box_ltwh[0]
        if padded_box_ltwh[1] + padded_box_ltwh[3] >= world_height:
            padded_box_ltwh[3] = world_height - padded_box_ltwh[1]

    return padded_box_ltwh


def apply_preprocess(image, bbox, padding_type, padding_value):
    padded_bbox = create_padding_bbox(
        box_ltwh=[int(v) for v in bbox],
        pad_type=padding_type,
        pad_value=padding_value,
        source_image_shape_hw=image.shape,
        allow_exceed_image=False
    )
    return image[padded_bbox[1]:padded_bbox[1] + padded_bbox[3], padded_bbox[0]:padded_bbox[0] + padded_bbox[2]]