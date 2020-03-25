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
from scipy.spatial.distance import cdist
from typing import Tuple

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


LEVELS = 1
MAX_CORNERS = 3000
CORNERS_RADIUS = 7

feature_params = dict(
    qualityLevel=0.005,
    minDistance=5,
    blockSize=9
)

lk_params = dict(
    winSize=(15, 15),
    minEigThreshold=0.001,
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


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


class FrameCornerStorage:
    def __init__(self, builder: _CornerStorageBuilder):
        self.builder = builder
        self.corners_pos = []
        self.corners_radius = []
        self.corners_indices = []
        self.new_index = 0
        self.frames = {}

    def reset(self):
        self.corners_pos = []
        self.corners_radius = []
        self.corners_indices = []

    def add_corners(self, corners: np.ndarray, radius: int, indices: np.ndarray):
        self.corners_pos.extend(corners)
        self.corners_radius.extend([radius] * len(corners))
        self.corners_indices.extend(indices)

    def move_to_frame(self, frame: int):
        corners = FrameCorners(
            np.array(self.corners_indices).astype(np.int32),
            np.array(self.corners_pos).astype(np.float32),
            np.array(self.corners_radius).astype(np.float32)
        )
        self.frames[frame] = corners
        self.builder.set_corners_at_frame(frame, corners)


def find_new_corners(image: np.ndarray, level: int) -> np.ndarray:
    image = cv2.resize(image, (image.shape[1] // 2**level, image.shape[0] // 2**level), interpolation=cv2.INTER_AREA)
    corners = cv2.goodFeaturesToTrack(image, maxCorners=MAX_CORNERS // 4**level, **feature_params, useHarrisDetector=False)
    return corners.reshape((-1, 2)).astype(np.float32)


def move_corners(corners: np.ndarray, image0: np.ndarray, image1: np.ndarray, level: int, prev_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image0 = (cv2.resize(image0, (image0.shape[1] // 2**level, image0.shape[0] // 2**level), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    image1 = (cv2.resize(image1, (image1.shape[1] // 2**level, image1.shape[0] // 2**level), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)

    new_corners, good, _ = cv2.calcOpticalFlowPyrLK(image0, image1, corners.astype(np.float32), None, **lk_params)

    good = good.reshape(-1)
    new_corners = new_corners[good == 1].astype(np.float32)
    new_indices = prev_indices[good == 1].astype(np.int32)

    return new_corners.reshape((-1, 2)), new_indices


def merge_corners(moved_corners: np.ndarray, new_corners: np.ndarray, moved_ids: np.ndarray, new_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dists = cdist(new_corners, moved_corners).min(axis=1)
    cond = dists >= feature_params['minDistance']
    new_corners = new_corners[cond]
    new_ids = new_ids[cond]

    final_corners = np.concatenate([moved_corners, new_corners])
    final_ids = np.concatenate([moved_ids.flatten(), new_ids.flatten()])
    return final_corners, final_ids


def _build_impl(frame_sequence: pims.FramesSequence, builder: _CornerStorageBuilder) -> None:
    frame_corners = FrameCornerStorage(builder)
    image_0 = frame_sequence[0]

    corners = []
    for level in range(LEVELS):
        corners.append(find_new_corners(image_0, level))
        indices = np.arange(frame_corners.new_index, frame_corners.new_index + len(corners[-1]))
        frame_corners.new_index += len(corners[-1])

        frame_corners.add_corners(corners[-1] * 2**level + 2**level // 2, CORNERS_RADIUS * 2**level, indices)
    frame_corners.move_to_frame(0)

    for frame, image in enumerate(frame_sequence[1:], 1):
        frame_corners.reset()

        next_corners = []
        for level in range(LEVELS):
            moved_corners, moved_ids = move_corners(corners[level], image_0, image, level, frame_corners.frames[frame - 1].ids)
            new_corners = find_new_corners(image, level)
            new_ids = np.arange(frame_corners.new_index, frame_corners.new_index + len(new_corners))
            frame_corners.new_index += len(new_corners)

            final_corners, final_indices = merge_corners(moved_corners, new_corners, moved_ids, new_ids)
            frame_corners.add_corners(final_corners * 2**level + 2**level // 2, CORNERS_RADIUS * 2**level, final_indices)
            next_corners.append(final_corners)

        frame_corners.move_to_frame(frame)

        image_0 = image
        corners = next_corners.copy()


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
