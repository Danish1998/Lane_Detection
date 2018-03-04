from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt

import configurations as cfg
import filters
import detector


def show_images(images, table_size, fig_size=(15, 8), cmap=None, titles=None, plot_title=None):
    """Shows images in table
    Args:
        images (list): list of input images
        table_size (tuple): (columns count, rows count)
        fig_size (tuple): picture (size x, size y) in inches
        cmap (list): list of cmap parameters for each image
        titles (list): list of images titles
        plot_title: title of the plot
    """
    rows = table_size[0]
    cols = table_size[1]
    fig, imgtable = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    for i in range(rows):
        for j in range(cols):
            img_index = i * cols + j
            if (isinstance(cmap, (list, tuple))):
                imgtable[i][j].imshow(images[img_index], cmap=cmap[i])
            else:
                img = images[img_index]
                if len(img.shape) == 3:
                    imgtable[i][j].imshow(img)
                else:
                    imgtable[i][j].imshow(img, cmap='gray')
            if not titles is None:
                imgtable[i][j].set_title(titles[img_index], fontsize=24)

    fig.set_title(plot_title, fontsize=28)
    fig.tight_layout()
    plt.show()


def get_desired_size(img):
    """Resizes and Crops an image to the desired size
    Args:
        img: image to be resized and cropped
    Return:
        cropped_img: image with desired size
    """
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise Exception("Image doesnot have any pixels")
    if img.shape[0] < cfg.smallest_size[0] or img.shape[1] < cfg.smallest_size[1]:
        raise Exception("Image is too small to scale")

    scale = max((cfg.desired_image_size[0] / img.shape[0]), (cfg.desired_image_size[1] / img.shape[1]))
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    start_x, end_x = int((scaled_img.shape[1] - cfg.desired_image_size[1]) / 2), int(
        (scaled_img.shape[1] + cfg.desired_image_size[1]) / 2)
    start_y, end_y = int((scaled_img.shape[0] - cfg.desired_image_size[0]) / 2), int(
        (scaled_img.shape[0] + cfg.desired_image_size[0]) / 2)
    cropped_img = scaled_img[start_y:end_y, start_x:end_x]
    return cropped_img


def calc_warp_points():
    """Calculates Source and Destination points
    Returns:
        src: source points
        dst: destination points
    """
    src = np.float32([[400, 720],
                      [600, 400],
                      [880, 400],
                      [1100, 720]])
    dst = np.float32([[180, 720],
                      [100, 0],
                      [1800, 0],
                      [800, 720]])
    return src, dst


def calc_tranform(src, dst):
    """Calculates Perspective and Inverse Perspective Transform Matrices
    Args:
        src: Source points
        dst: Destination Points
    Returns:
        Perspective Matrix and Inverse Perspective Transform Matrix
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp(img, M):
    """Warps the input image to change perspective
    Args:
        img: input image
        M: transformation matrix
    Returns:
        warped: transformed image
    """
    warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]),
                                      flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return warped_img


def process_img(img, old_left_x=None, old_right_x=None):
    img = get_desired_size(img)
    src, dst = calc_warp_points()
    M, Minv = calc_tranform(src, dst)
    warped_img = warp(img, M)
    binary_warped = filters.image_threshold(warped_img)

    roi = np.ones_like(warped_img) * 255
    roi_unwarped = warp(roi, Minv)

    if (old_left_x is None) or (old_right_x is None):
        lane, left_x, right_x = detector.blind_search(binary_warped)
    else:
        lane, left_x, right_x = detector.limited_search(binary_warped, old_left_x, old_right_x)

    unwarped_lane = warp(lane, Minv)
    img_ = cv2.addWeighted(img, 1, roi_unwarped, 0.2, 0)
    out_img = cv2.addWeighted(img_, 1, unwarped_lane, 0.6, 0)

    return out_img  # , left_x, right_x