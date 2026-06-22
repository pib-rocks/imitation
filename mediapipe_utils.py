import numpy as np
from collections import namedtuple
from math import ceil, sqrt, sin, cos, gcd


class HandRegion:
    """Stores detected hand information from the Edge pipeline."""

    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score
        self.pd_box = pd_box
        self.pd_kps = pd_kps


SSDAnchorOptions = namedtuple('SSDAnchorOptions', [
    'num_layers',
    'min_scale',
    'max_scale',
    'input_size_height',
    'input_size_width',
    'anchor_offset_x',
    'anchor_offset_y',
    'strides',
    'aspect_ratios',
    'reduce_boxes_in_lowest_layer',
    'interpolated_scale_aspect_ratio',
    'fixed_anchor_size',
])


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)


def generate_anchors(options):
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(
                options.min_scale, options.max_scale, last_same_stride_layer, n_strides
            )
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides - 1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(
                            options.min_scale, options.max_scale,
                            last_same_stride_layer + 1, n_strides
                        )
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1

        for i, r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                    else:
                        new_anchor = [
                            x_center, y_center,
                            anchor_width[anchor_id], anchor_height[anchor_id],
                        ]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer
    return np.array(anchors)


def generate_handtracker_anchors(input_size_width, input_size_height):
    anchor_options = SSDAnchorOptions(
        num_layers=4,
        min_scale=0.1484375,
        max_scale=0.75,
        input_size_height=input_size_height,
        input_size_width=input_size_width,
        anchor_offset_x=0.5,
        anchor_offset_y=0.5,
        strides=[8, 16, 16, 16],
        aspect_ratios=[1.0],
        reduce_boxes_in_lowest_layer=False,
        interpolated_scale_aspect_ratio=1.0,
        fixed_anchor_size=True,
    )
    return generate_anchors(anchor_options)


def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    p0x = cx - a * h - b * w
    p0y = cy + b * h - a * w
    p1x = cx + a * h - b * w
    p1y = cy - b * h - a * w
    p2x = int(2 * cx - p0x)
    p2y = int(2 * cy - p0y)
    p3x = int(2 * cx - p1x)
    p3y = int(2 * cy - p1y)
    return [[int(p0x), int(p0y)], [int(p1x), int(p1y)], [p2x, p2y], [p3x, p3y]]


def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size and corresponding setIspScale() parameters.
    """
    if size < 288:
        size = 288

    width, height = resolution

    if is_height:
        reference = height
        other = width
    else:
        reference = width
        other = height
    size_candidates = {}
    for s in range(288, reference, 16):
        f = gcd(reference, s)
        n = s // f
        d = reference // f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)

    min_dist = -1
    candidate = size
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1 or dist <= min_dist:
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]
