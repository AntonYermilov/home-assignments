#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import sys
import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)


def extract_points(corners1, corners2, view_mat1, view_mat2, intrinsic_mat, ids_to_remove=None):
    corr = build_correspondences(corners1, corners2, ids_to_remove)
    if not len(corr.ids):
        return None

    points3d, corr_ids, median_cos = triangulate_correspondences(
        corr,
        view_mat1,
        view_mat2,
        intrinsic_mat,
        TriangulationParameters(
            max_reprojection_error=1,
            min_triangulation_angle_deg=1,
            min_depth=0.1
        )
    )
    if not len(corr_ids):
        return None

    return corr_ids, points3d


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats = np.zeros((len(corner_storage), 3, 4), dtype=np.float64)
    processed_frames = np.zeros(len(corner_storage), dtype=np.bool)
    points3d = np.zeros((corner_storage.max_corner_id() + 1, 3), dtype=np.float64)
    added_points = np.zeros(corner_storage.max_corner_id() + 1, dtype=np.bool)

    print('Trying to extract 3d points from known frames...')

    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    processed_frames[known_view_1[0]] = processed_frames[known_view_2[0]] = True
    extracted_points = extract_points(
        corner_storage[known_view_1[0]],
        corner_storage[known_view_2[0]],
        view_mats[known_view_1[0]],
        view_mats[known_view_2[0]],
        intrinsic_mat
    )
    if not extracted_points:
        print('Extracting 3d points from common frames failed: '
              'either there are no common points, or triangulation angle between frames is too small.\n'
              'Try to choose another initial frames.', file=sys.stderr)
        exit(0)
    print('Succeeded! Trying to build point cloud...')

    added_points[extracted_points[0]] = True
    points3d[extracted_points[0]] = extracted_points[1]

    was_updated = True
    while was_updated:
        was_updated = False
        unprocessed_frames = np.arange(len(corner_storage), dtype=np.int32)[~processed_frames]
        for frame in unprocessed_frames:
            points3d_ids = np.arange(corner_storage.max_corner_id() + 1, dtype=np.int32)[added_points]
            common, (indices1, indices2) = snp.intersect(points3d_ids, corner_storage[frame].ids.flatten(), indices=True)
            if len(common) <= 5:
                continue

            try:
                print(f'Processing frame {frame}: ', end='')
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=points3d[common],
                    imagePoints=corner_storage[frame].points[indices2],
                    cameraMatrix=intrinsic_mat,
                    distCoeffs=None
                )
            except:
                retval = False

            if not retval:
                print(f'failed to solve PnP RANSAC, trying another frame')
                continue

            print(f'succeeded, found {len(inliers)} inliers')
            was_updated = True
            view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            for processed_frame in np.arange(len(corner_storage), dtype=np.int32)[processed_frames]:
                extracted_points = extract_points(
                    corner_storage[frame],
                    corner_storage[processed_frame],
                    view_mats[frame],
                    view_mats[processed_frame],
                    intrinsic_mat,
                    points3d_ids
                )

                if extracted_points:
                    added_points[extracted_points[0]] = True
                    points3d[extracted_points[0]] = extracted_points[1]
            processed_frames[frame] = True
            print(f'Current point cloud contains {sum(added_points)} points')

    for _ in range(2):
        for i in range(1, len(corner_storage)):
            if not processed_frames[i]:
                processed_frames[i] = True
                view_mats[i] = view_mats[i - 1]
        processed_frames = processed_frames[::-1]
        view_mats = view_mats[::-1]

    print(f'Finished building point cloud, now it contains {sum(added_points)} points')

    point_cloud_builder = PointCloudBuilder(
        ids=np.arange(corner_storage.max_corner_id() + 1, dtype=np.int32)[added_points],
        points=points3d[added_points]
    )

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
