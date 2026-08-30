# ====================================================================================
# Student implementation scaffold for the ENN583 2026 coding project assessment.
# ====================================================================================
# 
# You may add helper functions, classes, and additional modules inside ``src/``.
# IMPORTANT: Do not change the names or parameters of the three provided functions. 
# Gradescope and the local assessment checker call them directly. If you make changes to these
# functions or their parameters, your code will not run correctly on Gradescope and you will lose marks.
# 
# ``match_features(img_i, img_j)``
#     Match visual features between two provided images and write
#     ``results_matches.csv``.
# 
# ``estimate_relative_pose(dataset, frame_i, frame_j)``
#     Estimate the relative motion between two dataset frames and write
#     ``results_relative_pose.csv``.
# 
# ``visual_odometry(dataset)``
#     Estimate the trajectory for a dataset sequence and write
#     ``results_visual_odometry.csv``.


import csv

import numpy as np
import spatialmath as sm
import cv2 as cv


# ====================================================================================
# Your code goes here! 
# ====================================================================================
# Add your helper functions and classes below this line. 
# You can also create additional files in ``src/`` and import them here. 
#
# You may use packages provided by environment.yml. Ask the teaching team before
# adding another dependency because it may not be installed in Gradescope.
#
# Complete the three provided functions below. 
# Do not change their names or parameters because Gradescope calls them directly.
# Start with the match_features function, then implement estimate_relative_pose, and finally implement visual_odometry.
# You can and should reuse the functions, i.e. call match_features from estimate_relative_pose, and call estimate_relative_pose from visual_odometry.


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a KITTI image to the format expected by OpenCV features."""
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[..., 0]
        elif image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
        else:
            raise ValueError("Images must have 1, 3, or 4 channels")
    elif image.ndim != 2:
        raise ValueError("Images must be grayscale or colour arrays")

    if image.dtype != np.uint8:
        values = np.nan_to_num(image.astype(np.float32), copy=False)
        if values.size and 0.0 <= values.min() and values.max() <= 1.0:
            values *= 255.0
        image = np.clip(values, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _mutual_ratio_matches(descriptors_a, descriptors_b, norm, ratio):
    """Return unambiguous descriptor matches indexed by the first image."""
    if (descriptors_a is None or descriptors_b is None
            or len(descriptors_a) < 2 or len(descriptors_b) < 2):
        return {}

    matcher = cv.BFMatcher(norm)

    def one_way(a, b):
        result = {}
        for neighbours in matcher.knnMatch(a, b, k=2):
            if (len(neighbours) == 2
                    and neighbours[0].distance < ratio * neighbours[1].distance):
                result[neighbours[0].queryIdx] = neighbours[0]
        return result

    forward = one_way(descriptors_a, descriptors_b)
    reverse = one_way(descriptors_b, descriptors_a)
    return {
        query: match for query, match in forward.items()
        if match.trainIdx in reverse
        and reverse[match.trainIdx].trainIdx == query
    }


def _relative_pose_from_stereo(dataset, frame_i: int, frame_j: int) -> sm.SE3:
    """Estimate T_i_j (the physical pose of camera j expressed in camera i)."""
    left_i, right_i = dataset.stereo(frame_i)
    left_j, _ = dataset.stereo(frame_j)
    images = [_gray_uint8(image) for image in (left_i, right_i, left_j)]

    if hasattr(cv, "SIFT_create"):
        detector = cv.SIFT_create(
            nfeatures=7000, contrastThreshold=0.015, edgeThreshold=12
        )
        norm, ratio = cv.NORM_L2, 0.78
    else:
        detector = cv.ORB_create(nfeatures=7000, fastThreshold=8)
        norm, ratio = cv.NORM_HAMMING, 0.82

    features = [detector.detectAndCompute(image, None) for image in images]
    (key_left_i, desc_left_i), (key_right_i, desc_right_i), \
        (key_left_j, desc_left_j) = features

    stereo = _mutual_ratio_matches(desc_left_i, desc_right_i, norm, ratio)
    temporal = _mutual_ratio_matches(desc_left_i, desc_left_j, norm, ratio)

    calibration_left = dataset.camera_calibration(camera=2)
    calibration_right = dataset.camera_calibration(camera=3)
    P_left = np.asarray(calibration_left["P"], dtype=np.float64)
    P_right = np.asarray(calibration_right["P"], dtype=np.float64)
    K = P_left[:, :3].copy()

    # A rectified projection matrix encodes its x camera centre as -P[0,3]/fx.
    centre_left = -P_left[0, 3] / P_left[0, 0]
    centre_right = -P_right[0, 3] / P_right[0, 0]
    baseline = float(centre_right - centre_left)
    if not np.isfinite(baseline) or abs(baseline) < 1e-6:
        raise ValueError("The stereo calibration does not contain a valid baseline")

    object_points = []
    image_points = []
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for query in stereo.keys() & temporal.keys():
        u_left, v_left = key_left_i[query].pt
        u_right, v_right = key_right_i[stereo[query].trainIdx].pt
        u_j, v_j = key_left_j[temporal[query].trainIdx].pt
        disparity = u_left - u_right

        # Reject matches inconsistent with rectified stereo geometry.
        if abs(v_left - v_right) > 2.0 or disparity * baseline <= 0.5 * abs(baseline):
            continue
        depth = fx * baseline / disparity
        if not np.isfinite(depth) or depth <= 1.0 or depth > 120.0:
            continue
        object_points.append([
            (u_left - cx) * depth / fx,
            (v_left - cy) * depth / fy,
            depth,
        ])
        image_points.append([u_j, v_j])

    if len(object_points) < 6:
        raise RuntimeError(
            f"Only {len(object_points)} valid stereo-temporal matches were found"
        )

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    success, rotation_vector, translation, inliers = cv.solvePnPRansac(
        object_points, image_points, K, None,
        iterationsCount=3000, reprojectionError=2.5, confidence=0.999,
        flags=cv.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP could not estimate a reliable relative pose")

    inlier_ids = inliers.ravel()
    success, rotation_vector, translation = cv.solvePnP(
        object_points[inlier_ids], image_points[inlier_ids], K, None,
        rotation_vector, translation, True, flags=cv.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("PnP pose refinement failed")

    rotation_i_to_j_coordinates, _ = cv.Rodrigues(rotation_vector)
    coordinate_transform = np.eye(4)
    coordinate_transform[:3, :3] = rotation_i_to_j_coordinates
    coordinate_transform[:3, 3] = translation.ravel()

    # solvePnP maps fixed-point coordinates from camera i to camera j. The
    # requested physical pose of camera j in camera i is its inverse.
    return sm.SE3(coordinate_transform, check=False).inv()



#  ====================================================================================
#  ====================================================================================
def match_features(img_i: np.ndarray, img_j: np.ndarray):
    """Detect and match visual features between two provided images.

    This is the first, most local part of the visual odometry pipeline. Your
    implementation should detect keypoints in ``img_i`` and ``img_j``, compute
    descriptors, match the descriptors, and apply any filtering strategy you
    think is appropriate.

    Parameters
    ----------
    img_i : np.ndarray
        First image. It may be grayscale or colour. Pixel coordinates u_i,v_i written to
        the output file must refer to this image's coordinate system.

    img_j : np.ndarray
        Second image. It may be grayscale or colour. Pixel coordinates u_j,v_j written
        to the output file must refer to this image's coordinate system.

    Returns
    -------
    None
        The assessment does not require this function to return anything.

    Side effects
    ------------
    Write a CSV file named ``results_matches.csv`` in the current working
    directory. Each row should describe one matched point pair between the two
    input images. The required columns are:

    ``match_id,u_i,v_i,u_j,v_j``
    
    Here ``u_i,v_i`` are the pixel coordinates of the feature in ``img_i`` and
    ``u_j,v_j`` are the pixel coordinates of the corresponding feature in
    ``img_j``. The ``match_id`` column should contain a unique integer for each
    match, starting from 0. The order of the features does not matter.

    Example
    -------
    The local checker and Gradescope call this function like this:

    ```python
    img_i = dataset.stereo(frame_i)[0]  # left image from frame_i
    img_j = dataset.stereo(frame_j)[0]  # left image from frame_j

    match_features(img_i, img_j)
    ```
    
    """ 

    def as_gray_uint8(image: np.ndarray) -> np.ndarray:
        """Return an OpenCV-compatible grayscale image without changing geometry."""
        image = np.asarray(image)
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[..., 0]
            elif image.shape[2] == 3:
                # Whether the input is RGB or BGR has negligible influence on
                # local features, but KITTI loaders normally return RGB arrays.
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
            else:
                raise ValueError("Images must have 1, 3, or 4 channels")
        elif image.ndim != 2:
            raise ValueError("Images must be two-dimensional or colour arrays")

        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        values = np.nan_to_num(image.astype(np.float32), copy=False)
        if values.size == 0:
            return np.empty(values.shape, dtype=np.uint8)
        low, high = float(values.min()), float(values.max())
        # Preserve the conventional [0, 1] and [0, 255] intensity scales.
        if 0.0 <= low and high <= 1.0:
            values = values * 255.0
        elif low < 0.0 or high > 255.0:
            values = cv.normalize(values, None, 0, 255, cv.NORM_MINMAX)
        return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))

    # Convert images to greyscale uint8 arrays for OpenCV feature detection and matching.
    gray_i, gray_j = as_gray_uint8(img_i), as_gray_uint8(img_j)

    # SIFT is robust to the modest scale and viewpoint changes between KITTI
    # frames. ORB keeps the function usable with OpenCV builds lacking SIFT.
    
    # Choose SIFT if available, otherwise fall back to ORB. The parameters below
    # are tuned for the KITTI dataset.
    if hasattr(cv, "SIFT_create"):
        detector = cv.SIFT_create(nfeatures=5000, contrastThreshold=0.02,
                                  edgeThreshold=12)
        norm = cv.NORM_L2
        ratio = 0.75
    else:
        detector = cv.ORB_create(nfeatures=5000, fastThreshold=10)
        norm = cv.NORM_HAMMING
        ratio = 0.80

    # Detect keypoints and compute descriptors for both images.
    keypoints_i, descriptors_i = detector.detectAndCompute(gray_i, None)
    keypoints_j, descriptors_j = detector.detectAndCompute(gray_j, None)
    matches = []

    # Check that both images have enough keypoints to match. If either image has
    # fewer than two keypoints, the matcher will fail. In that case, we return
    # an empty match list, which is still a valid output.
    if (descriptors_i is not None and descriptors_j is not None
            and len(descriptors_i) >= 2 and len(descriptors_j) >= 2):
        
        # Create brute force matcher
        matcher = cv.BFMatcher(norm)

        def ratio_matches(first, second):
            accepted = {}
            
            # Find two nearest matches
            for neighbours in matcher.knnMatch(first, second, k=2):
                if len(neighbours) == 2 and neighbours[0].distance < ratio * neighbours[1].distance:
                    accepted[neighbours[0].queryIdx] = neighbours[0]
            return accepted

        # Forward and reverse matching
        forward = ratio_matches(descriptors_i, descriptors_j) # image i to image j
        reverse = ratio_matches(descriptors_j, descriptors_i) # image j to image i
        
        # Mutual matching removes many ambiguous correspondences on repeated
        # road markings, windows, vegetation, and image borders.
        matches = [m for query, m in forward.items()
                   if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == query]
        
        # Sort by distance so that the best matches are first.
        matches.sort(key=lambda m: m.distance)

        # Use epipolar geometry as a final outlier rejection step. If geometry
        # cannot be estimated, the descriptor-filtered matches remain useful.
        
        # need at least 8 matches to compute the fundamental matrix
        if len(matches) >= 8:
            points_i = np.float32([keypoints_i[m.queryIdx].pt for m in matches])
            points_j = np.float32([keypoints_j[m.trainIdx].pt for m in matches])
            
            # Estimate the fundamental matrix using RANSAC to filter out outliers.
            _, mask = cv.findFundamentalMat(
                points_i, points_j, cv.FM_RANSAC, 1.5, 0.999, 5000
            )
            # Keep only the inlier matches that are consistent with the estimated epipolar lines
            if mask is not None and mask.size == len(matches):
                inliers = [m for m, keep in zip(matches, mask.ravel()) if keep]
                if len(inliers) >= 8:
                    matches = inliers
    # Write the matches to a CSV file in the required format.
    with open("results_matches.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["match_id", "u_i", "v_i", "u_j", "v_j"])
        for match_id, match in enumerate(matches):
            u_i, v_i = keypoints_i[match.queryIdx].pt
            u_j, v_j = keypoints_j[match.trainIdx].pt
            writer.writerow([match_id, u_i, v_i, u_j, v_j])

    return None




#  ====================================================================================
#  ====================================================================================
def estimate_relative_pose(dataset, frame_i: int, frame_j: int):
    """Estimate the relative camera pose between two dataset frames.

    This is the frame-to-frame motion-estimation stage of the visual odometry
    pipeline. Your implementation should load the two requested frames from the
    dataset, establish visual correspondences, and estimate the rigid
    transformation from ``frame_i`` to ``frame_j``.
    
    Parameters
    ----------
    dataset:
        Dataset supplied by the runner. It provides ``stereo(i)``,
        ``camera_calibration(camera)``, ``frame_count``, and ``len(dataset)``.
        
        While you are developing your code, you can also access ``ground_truth_pose(i)`` to get the true pose of frame ``i``. 
        However, ground-truth poses are unavailable during marking.

    frame_i : int
        Index of the first frame.

    frame_j : int
        Index of the second frame.

    Returns
    -------
    None
        The assessment does not require this function to return anything.

    Side effects
    ------------
    Write a CSV file named ``results_relative_pose.csv`` in the current working
    directory. The file should contain your estimated relative pose. The
    required format is one row with these columns:

    ``frame_i,frame_j,x,y,z,roll,pitch,yaw``

    Here ``x,y,z`` are the translation components and ``roll,pitch,yaw`` are
    Euler angles, all describing the left-camera pose of ``frame_j`` relative
    to the left-camera pose of ``frame_i``. Angles must be in radians. The
    checker interprets them using SpatialMath's
    ``SE3.RPY(roll, pitch, yaw, order="zyx")`` convention.

    Example
    -------
    The local checker and Gradescope call this function like this:

    ```python
    frame_i = 0
    frame_j = 1

    estimate_relative_pose(dataset, frame_i, frame_j)
    ```
       
    """

    frame_i, frame_j = int(frame_i), int(frame_j)
    frame_count = len(dataset)
    if not (0 <= frame_i < frame_count and 0 <= frame_j < frame_count):
        raise IndexError("frame_i and frame_j must be valid dataset frame indices")

    pose = _relative_pose_from_stereo(dataset, frame_i, frame_j)
    roll, pitch, yaw = pose.rpy(order="zyx", unit="rad")

    with open("results_relative_pose.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["frame_i", "frame_j", "x", "y", "z",
                         "roll", "pitch", "yaw"])
        writer.writerow([
            frame_i, frame_j,
            *[float(value) for value in pose.t],
            float(roll), float(pitch), float(yaw),
        ])

    return None

#  ====================================================================================
#  ====================================================================================
def visual_odometry(dataset):
    """Estimate the camera trajectory for a full dataset sequence.

    This is the complete visual odometry stage. Your implementation should
    estimate the camera pose for each frame in the sequence, usually by chaining
    together frame-to-frame relative poses.

    Parameters
    ----------
    dataset:
        Dataset supplied by the runner. It provides ``stereo(i)``,
        ``camera_calibration(camera)``, ``frame_count``, and ``len(dataset)``.
        
        While you are developing your code, you can also access ``ground_truth_pose(i)`` to get the true pose of frame ``i``. 
        However, ground-truth poses are unavailable during marking.

    Returns
    -------
    None
        The assessment does not require this function to return anything.

    Side effects
    ------------
    Write a CSV file named ``results_visual_odometry.csv`` in the current
    working directory. The file should contain your estimated trajectory. The
    required format is one row per frame with these columns:

    ``frame,x,y,z,roll,pitch,yaw``

    Each pose should describe the camera pose for that frame relative to frame
    0. The pose for frame 0 should normally be the identity pose.

    Example
    -------
    The local checker and Gradescope call this function like this:

    ```python
    visual_odometry(dataset)
    ```

    """

    frame_count = len(dataset)
    if frame_count < 1:
        raise ValueError("The dataset must contain at least one frame")

    # T_0_k is the physical pose of camera k expressed in camera 0. If
    # T_k_(k+1) is expressed in camera k, ordinary SE(3) composition gives the
    # next trajectory pose: T_0_(k+1) = T_0_k @ T_k_(k+1).
    poses = [sm.SE3()]
    for frame in range(1, frame_count):
        relative_pose = _relative_pose_from_stereo(dataset, frame - 1, frame)
        poses.append(poses[-1] @ relative_pose)

    with open("results_visual_odometry.csv", "w", newline="",
              encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["frame", "x", "y", "z", "roll", "pitch", "yaw"])
        for frame, pose in enumerate(poses):
            roll, pitch, yaw = pose.rpy(order="zyx", unit="rad")
            writer.writerow([
                frame,
                *[float(value) for value in pose.t],
                float(roll), float(pitch), float(yaw),
            ])

    return None
