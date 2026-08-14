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

    gray_i, gray_j = as_gray_uint8(img_i), as_gray_uint8(img_j)

    # SIFT is robust to the modest scale and viewpoint changes between KITTI
    # frames. ORB keeps the function usable with OpenCV builds lacking SIFT.
    if hasattr(cv, "SIFT_create"):
        detector = cv.SIFT_create(nfeatures=5000, contrastThreshold=0.02,
                                  edgeThreshold=12)
        norm = cv.NORM_L2
        ratio = 0.75
    else:
        detector = cv.ORB_create(nfeatures=5000, fastThreshold=10)
        norm = cv.NORM_HAMMING
        ratio = 0.80

    keypoints_i, descriptors_i = detector.detectAndCompute(gray_i, None)
    keypoints_j, descriptors_j = detector.detectAndCompute(gray_j, None)
    matches = []

    if (descriptors_i is not None and descriptors_j is not None
            and len(descriptors_i) >= 2 and len(descriptors_j) >= 2):
        matcher = cv.BFMatcher(norm)

        def ratio_matches(first, second):
            accepted = {}
            for neighbours in matcher.knnMatch(first, second, k=2):
                if len(neighbours) == 2 and neighbours[0].distance < ratio * neighbours[1].distance:
                    accepted[neighbours[0].queryIdx] = neighbours[0]
            return accepted

        forward = ratio_matches(descriptors_i, descriptors_j)
        reverse = ratio_matches(descriptors_j, descriptors_i)
        # Mutual matching removes many ambiguous correspondences on repeated
        # road markings, windows, vegetation, and image borders.
        matches = [m for query, m in forward.items()
                   if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == query]
        matches.sort(key=lambda m: m.distance)

        # Use epipolar geometry as a final outlier rejection step. If geometry
        # cannot be estimated, the descriptor-filtered matches remain useful.
        if len(matches) >= 8:
            points_i = np.float32([keypoints_i[m.queryIdx].pt for m in matches])
            points_j = np.float32([keypoints_j[m.trainIdx].pt for m in matches])
            _, mask = cv.findFundamentalMat(
                points_i, points_j, cv.FM_RANSAC, 1.5, 0.999, 5000
            )
            if mask is not None and mask.size == len(matches):
                inliers = [m for m, keep in zip(matches, mask.ravel()) if keep]
                if len(inliers) >= 8:
                    matches = inliers

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

    # ====================================================================================
    # Implement frame-to-frame motion estimation here.
    #
    # Suggested steps:
    #   1. Load the left images for frame_i and frame_j using dataset.stereo(...).
    #   2. Find feature matches between the two frames.
    #   3. Use calibration/depth/geometry to estimate the relative pose.
    #   4. Write results_relative_pose.csv with columns:
    #      frame_i,frame_j,x,y,z,roll,pitch,yaw
    # ====================================================================================

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

    # ====================================================================================
    # Implement full visual odometry here.
    #
    # Suggested steps:
    #   1. Start with the identity pose for frame 0.
    #   2. Estimate relative poses between successive frames.
    #   3. Chain the relative poses to build the full trajectory.
    #   4. Write results_visual_odometry.csv with columns:
    #      frame,x,y,z,roll,pitch,yaw
    # ====================================================================================

    return None
