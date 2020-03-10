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
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    _remove_correspondences_with_ids
)


TRIANGULATION_PARAMS = TriangulationParameters(
    max_reprojection_error=3,
    min_triangulation_angle_deg=1.5,
    min_depth=0.025
)


def initialize_views(intrinsic_mat, corner_storage):
    best_frame = None
    pi_1 = eye3x4()
    for frame0 in [0]:
        for frame in range(frame0 + 5, len(corner_storage) - 5):
            corr = build_correspondences(corner_storage[frame0], corner_storage[frame])
            essential_mat, mask = cv2.findEssentialMat(corr.points_1, corr.points_2, intrinsic_mat, method=cv2.RANSAC)
            if essential_mat is None or mask is None:
                continue

            mask = mask.flatten()
            R1, R2, t = cv2.decomposeEssentialMat(essential_mat)

            pi_2s = [np.hstack((R1, t)), np.hstack((R1, -t)), np.hstack((R2, t)), np.hstack((R2, -t))]
            for pi_2 in pi_2s:
                _corr = _remove_correspondences_with_ids(corr, np.arange(len(mask), dtype=np.int32)[mask == 0])
                points3d, _, _ = triangulate_correspondences(_corr, pi_1, pi_2, intrinsic_mat, TRIANGULATION_PARAMS)
                if not best_frame or len(points3d) > best_frame[0]:
                    best_frame = (len(points3d), frame, pi_2.copy(), frame0)

    print(f'frame0={best_frame[3]}, frame1={best_frame[1]}')
    return (best_frame[3], view_mat3x4_to_pose(pi_1)), (best_frame[1], view_mat3x4_to_pose(best_frame[2]))


def extract_points(corners1, corners2, view_mat1, view_mat2, intrinsic_mat, ids_to_remove=None):
    corr = build_correspondences(corners1, corners2, ids_to_remove)
    if not len(corr.ids):
        return None

    points3d, corr_ids, median_cos = triangulate_correspondences(
        corr,
        view_mat1,
        view_mat2,
        intrinsic_mat,
        TRIANGULATION_PARAMS
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
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = initialize_views(intrinsic_mat, corner_storage)

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
        print('Extracting 3d points from known frames failed: '
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
        for i in range(1, len(view_mats)):
            if not processed_frames[i] and processed_frames[i - 1]:
                processed_frames[i] = True
                view_mats[i] = view_mats[i - 1]
        processed_frames = processed_frames[::-1]
        view_mats = view_mats[::-1]

    dists = []
    for i in range(1, len(corner_storage)):
        dists.append(np.linalg.norm(view_mats[i] - view_mats[i - 1]))
    dists = np.array(dists)

    max_dist = np.median(dists) * 10

    for i in range(1, len(corner_storage)):
        if dists[i - 1] > max_dist:
            j = i
            while j < len(corner_storage) and dists[j - 1] > max_dist:
                j += 1
            for k in range(i, j - 1):
                R = view_mats[i - 1][:, :3] if k - i + 1 <= j - k - 1 and j != len(corner_storage) else view_mats[j - 1][:, :3]
                t1 = view_mats[i - 1][:, 3]
                t2 = view_mats[j - 1][:, 3] if j != len(corner_storage) else view_mats[i - 1][:, 3]
                t = t1 + (k - i + 1) * (t2 - t1) / (j - i)
                view_mats[k] = np.hstack((R, t.reshape(3, 1)))

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
