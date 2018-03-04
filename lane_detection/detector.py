from __future__ import division


import numpy as np
import cv2

import configurations as cfg


def line_sanity(fit, x, y, old_x):
    if fit is None:
        status = False
    else:
        y_ = np.linspace(0, cfg.desired_image_size[0] - 1, cfg.desired_image_size[0])
        new_x = get_intercepts(fit, y)
        margin = 150
        delta_x = np.abs(new_x - old_x)
        if len(delta_x > margin) == 100:
            status = True
        else:
            status = False
    return status


def lane_sanity(left_fit, right_fit, left_x, right_x, left_y, right_y, old_left_x, old_right_x):
    y_ = np.linspace(0, cfg.desired_image_size[0] - 1, cfg.desired_image_size[0])
    new_left_x = get_intercepts(left_fit, y_)
    new_right_x = get_intercepts(right_fit, y_)

    lane_width = 700
    delta_x = new_right_x - new_left_x
    if np.min(delta_x) > 0 and np.max(delta_x) < lane_width:
        status = True
    else:
        status = False
    return status


def predict_left_line(left_x, left_y, old_left_x, right_fit):
    lane_width = 700
    y_ = np.linspace(0, cfg.desired_image_size[0] - 1, cfg.desired_image_size[0])
    new_left_x = get_intercepts(right_fit, y_) - lane_width

    margin = 80
    new_left_x[(new_left_x - old_left_x) > margin] = old_left_x[(new_left_x - old_left_x) > margin] + margin
    new_left_x[(old_left_x - new_left_x) > margin] = old_left_x[(old_left_x - new_left_x) > margin] - margin

    left_fit = fit_line(new_left_x, y_)
    return left_fit


def predict_right_line(right_x, right_y, old_right_x, left_fit):
    lane_width = 700
    y_ = np.linspace(0, cfg.desired_image_size[0] - 1, cfg.desired_image_size[0])
    new_right_x = get_intercepts(left_fit, y_) + lane_width

    margin = 80
    new_right_x[(new_right_x - old_right_x) > margin] = old_right_x[(new_right_x - old_right_x) > margin] + margin
    new_right_x[(old_right_x - new_right_x) > margin] = old_right_x[(old_right_x - new_right_x) > margin] - margin

    right_fit = fit_line(new_right_x, y_)
    return right_fit


def predict_lane(left_x, right_x, left_y, right_y, old_left_x, old_right_x):
    y_ = np.linspace(0, cfg.desired_image_size[0] - 1, cfg.desired_image_size[0])
    left_fit = fit_line(old_left_x, y_)
    right_fit = fit_line(old_right_x, y_)

    return left_fit, right_fit


# Should add sanity checks
def fit_line(x, y, degree=2):
    if len(x) == 0:
        return None
    return np.polyfit(y, x, deg=degree)


def get_intercepts(fit, y):
    """Get x intercepts for given y values and fit
    Args:
        fit: co-efficients of the fit
        y: y intercepts to fit
    Returns:
        x: x intercepts for given y
    """
    # Get x intercepts for given fit
    x = fit[0] * y * y + fit[1] * y + fit[2]
    # Avoid index values outside the image range
    x[x < 0] = 0
    x[x > cfg.desired_image_size[1]] = cfg.desired_image_size[1]
    # Return x intercepts
    return x


def check_and_fit(left_x, right_x, left_y, right_y, old_left_x=None, old_right_x=None):
    if not old_left_x:
        old_left_x = np.ones((cfg.desired_image_size[0], 1)) * 300
    if not old_right_x:
        old_right_x = np.ones((cfg.desired_image_size[0], 1)) * 1000

    left_fit = fit_line(left_x, left_y)
    right_fit = fit_line(right_x, right_y)

    left_status = line_sanity(left_fit, left_x, left_y, old_left_x)
    right_status = line_sanity(right_fit, right_x, right_y, old_right_x)
    lane_status = lane_sanity(left_fit, right_fit,
                              left_x, right_x,
                              left_y, right_y,
                              old_left_x, old_right_x)
    if left_status and right_status and lane_status:
        return left_fit, right_fit
    elif left_status and not right_status:
        right_fit = predict_left_line(right_x, right_y, old_right_x, left_fit)
        return left_fit, right_fit
    elif right_status and not left_status:
        left_fit = predict_right_line(left_x, left_y, old_left_x, right_fit)
        return left_fit, right_fit
    else:
        left_fit, right_fit = predict_lane(left_x, right_x,
                                           left_y, right_y,
                                           old_left_x, old_right_x)
        return left_fit, right_fit


def draw_lane(img, left_x, right_x, left_y, right_y):
    """Draw the Lane based on X, and Y points for Left and Right Lanes on Image
    Args:
        img: image
        left_x: x intercepts of left lane
        right_x: x intercepts of right lane
        left_y: y intercepts of left lane
        right_y: y intercepts of right lane
    Returns:
        img: image with the polygon
    """
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_x, left_y])))])
    pts_right = np.array([np.transpose(np.vstack([right_x, right_y]))])
    pts = np.hstack((pts_left, pts_right))
    img = cv2.polylines(img, np.int_(pts), isClosed=False, color=(0, 0, 255), thickness=25)
    img = cv2.fillPoly(img, np.int_(pts), (34, 255, 34))
    return img


def blind_search(binary_warped, x_offset=-150):
    # Take a histogram of bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Starting point for left and right lines
    midpoint = int(cfg.desired_image_size[1] / 2) + x_offset
    leftx_base = np.argmax(histogram[midpoint - 400:midpoint]) + midpoint - 400
    rightx_base = np.argmax(histogram[midpoint:midpoint + 400]) + midpoint
    # number and height and width of windows
    n_windows = 8
    window_height = int(cfg.desired_image_size[0] / n_windows)
    margin = 80
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current position for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set minimum number of pixels to recenter window
    minpix = 40
    # Indices for left and right lanes
    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                          & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                           & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_idx = np.concatenate(left_lane_inds)
    right_lane_idx = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_x = nonzerox[left_lane_idx]
    left_y = nonzeroy[left_lane_idx]
    right_x = nonzerox[right_lane_idx]
    right_y = nonzeroy[right_lane_idx]

    # Fit a second order curve to each line
    left_fit = fit_line(left_x, left_y)
    right_fit = fit_line(right_x, right_y)
    y_ = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fit_x = get_intercepts(left_fit, y_)
    right_fit_x = get_intercepts(right_fit, y_)

    # Draw a polygon over the lane
    out_img = np.zeros_like(binary_warped)
    out_img = np.dstack((out_img, out_img, out_img))
    out_img = draw_lane(out_img, left_fit_x, right_fit_x, y_, y_)

    return out_img, left_fit_x, right_fit_x


# Should check if it works
def limited_search(binary_warped, old_left_x, old_right_x):
    # Set the margin of search
    margin = 100
    # Create a mask to filter out unwanted pixels
    left_lane_mask = np.zeros_like(binary_warped)
    right_lane_mask = np.zeros_like(binary_warped)
    for y in range(0, binary_warped.shape[0]):
        left_lane_mask[y:y + 1, int(old_left_x[y] - margin): int(old_left_x[y] + margin)] = np.ones((1, 2 * margin))
    for y in range(0, binary_warped.shape[0]):
        right_lane_mask[y:y + 1, int(old_right_x[y] - margin): int(old_right_x[y] + margin)] = np.ones((1, 2 * margin))
    # Filter out unwanted pixels
    left_lane = cv2.bitwise_and(binary_warped, left_lane_mask)
    right_lane = cv2.bitwise_and(binary_warped, right_lane_mask)
    # Identify the x and y positions of all nonzero pixels in the image
    left_lane_idx = left_lane.nonzero()
    right_lane_idx = right_lane.nonzero()

    # Extract left and right line pixel positions
    left_x = np.array(left_lane_idx[1])
    left_y = np.array(left_lane_idx[0])
    right_x = np.array(right_lane_idx[1])
    right_y = np.array(right_lane_idx[0])

    # Fit a second order curve to each line
    left_fit = fit_line(left_x, left_y)
    right_fit = fit_line(right_x, right_y)
    y_ = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fit_x = get_intercepts(left_fit, y_)
    right_fit_x = get_intercepts(right_fit, y_)

    # Draw a polygon over the lane
    out_img = np.zeros_like(binary_warped)
    out_img = np.dstack((out_img, out_img, out_img))
    out_img = draw_lane(out_img, left_fit_x, right_fit_x, y_, y_)

    return out_img, left_fit_x, right_fit_x