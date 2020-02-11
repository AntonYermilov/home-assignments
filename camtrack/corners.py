#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from typing import List

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


feature_params = dict(
    maxCorners=400,
    qualityLevel=0.01,
    minDistance=13,
    blockSize=9
)

corners_radius = feature_params['minDistance']

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

LEVELS = 6


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def add_corners(frame: int, corners: np.ndarray, builder: _CornerStorageBuilder):
    corners = FrameCorners(
        np.arange(len(corners)),
        corners,
        np.array([corners_radius] * len(corners))
    )
    builder.set_corners_at_frame(frame, corners)


def find_new_corners(image: np.ndarray, scale=1) -> np.ndarray:
    blockSize = feature_params['blockSize']
    image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, ksize=(blockSize, blockSize), sigmaX=0.5)
    corners = cv2.goodFeaturesToTrack(image, **feature_params, useHarrisDetector=False)
    corners = corners.squeeze().astype(int)
    return corners


def move_corners(corners: np.ndarray, image0: np.ndarray, image1: np.ndarray, scale=1) -> np.ndarray:
    image0 = cv2.resize(image0, (image0.shape[1] // scale, image0.shape[0] // scale), interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, (image1.shape[1] // scale, image1.shape[0] // scale), interpolation=cv2.INTER_AREA)
    image0 = (image0 * 255).astype(np.uint8)
    image1 = (image1 * 255).astype(np.uint8)
    new_corners, _, _ = cv2.calcOpticalFlowPyrLK(image0, image1, corners.astype(np.float32), None, **lk_params)
    new_corners = new_corners.astype(int)
    return new_corners.astype(int)


def merge_corners(moved_corners: np.ndarray, new_corners: np.ndarray, image, scale=1) -> np.ndarray:
    image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv2.INTER_AREA)
    h, w = image.shape
    blockSize = feature_params['blockSize']
    min_eigen_values = cv2.cornerMinEigenVal(image, blockSize=blockSize, ksize=3)
    max_eigen_value = np.max(min_eigen_values)

    merged_corners = []
    used = np.zeros(new_corners.shape[0]).astype(bool)
    for corner in moved_corners:
        if 0 <= corner[0] < w and 0 <= corner[1] < h:
            dist = np.sum((new_corners - corner)**2, axis=-1)
            ind = np.argmin(dist)
            if np.sqrt(dist[ind]) <= feature_params['minDistance']:
                corner = (corner + new_corners[ind]) // 2
                merged_corners.append(corner)
                used[ind] = True
            elif min_eigen_values[corner[1], corner[0]] >= feature_params['qualityLevel'] * max_eigen_value:
                merged_corners.append(corner)

    for corner in new_corners[~used]:
        if 0 <= corner[0] < w and 0 <= corner[1] < h:
            merged_corners.append(corner)

    merged_corners.sort(key=lambda k: -min_eigen_values[k[1], k[0]])
    merged_corners = np.array(merged_corners)

    final_corners = []
    used = np.zeros(merged_corners.shape[0]).astype(bool)
    for i, corner in enumerate(merged_corners):
        prefix = merged_corners[used]

        if len(prefix) == 0:
            final_corners.append(corner)
            used[i] = True
            continue

        dist = np.sqrt(np.sum((prefix - corner)**2, axis=-1))
        if np.min(dist) >= feature_params['minDistance']:
            final_corners.append(corner)
            used[i] = True

    if len(final_corners) > feature_params['maxCorners'] // scale:
        final_corners = final_corners[:feature_params['maxCorners'] // scale]
    return np.array(final_corners)


def merge_levels(corners: List[np.ndarray], image) -> np.ndarray:
    weights = []
    for level in range(LEVELS):
        blockSize = feature_params['blockSize']
        min_eigen_values = cv2.cornerMinEigenVal(image, blockSize=blockSize, ksize=3)

        if len(corners[level]) > feature_params['maxCorners'] // 2**level:
            corners[level] = corners[level][:feature_params['maxCorners'] // 2**level]

        weights.append(np.array([min_eigen_values[c[1], c[0]] for c in corners[level]]))
        corners[level] = corners[level] * 2**level + 2**level // 2

        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)

    merged_corners = []
    for level in range(LEVELS):
        merged_corners.extend(zip(corners[level], weights[level]))

    merged_corners.sort(key=lambda x: -x[1])
    merged_corners = np.array([x[0] for x in merged_corners])

    final_corners = []
    used = np.zeros(merged_corners.shape[0]).astype(bool)
    for i, corner in enumerate(merged_corners):
        prefix = merged_corners[used]

        if len(prefix) == 0:
            final_corners.append(corner)
            used[i] = True
            continue

        dist = np.sqrt(np.sum((prefix - corner) ** 2, axis=1))
        if np.min(dist) >= feature_params['minDistance']:
            final_corners.append(corner)
            used[i] = True

    return np.array(final_corners)


def _build_impl(frame_sequence: pims.FramesSequence, builder: _CornerStorageBuilder) -> None:
    global feature_params
    global corners_radius
    global LEVELS

    image_0 = frame_sequence[0]
    height, width = image_0.shape[0], image_0.shape[1]

    feature_params['minDistance'] = width // 140
    corners_radius = width // 140
    LEVELS = width // 350

    corners = []
    for level in range(LEVELS):
        corners.append(find_new_corners(image_0, scale=2**level))
    final_corners = merge_levels(corners.copy(), image_0)
    add_corners(0, final_corners, builder)

    for frame, image in enumerate(frame_sequence[1:], 1):
        moved_corners = []
        new_corners = []
        for level in range(LEVELS):
            moved_corners.append(move_corners(corners[level], image_0, image, scale=2**level))
            new_corners.append(find_new_corners(image, scale=2**level))

        corners = []
        for level in range(LEVELS):
            corners.append(merge_corners(moved_corners[level], new_corners[level], image, scale=2**level))
        final_corners = merge_levels(corners.copy(), image)

        add_corners(frame, final_corners, builder)
        image_0 = image


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
