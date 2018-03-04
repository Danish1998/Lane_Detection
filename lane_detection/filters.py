from __future__ import division

import numpy as np
import cv2


def morphology_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls_s = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
    src = 0.8 * hls_s + 0.2 * gray
    src = (src - np.min(src)) / (np.max(src) - np.min(src)) - 0.5

    blur_filter = np.zeros((1, 5))
    blur_filter.fill(1)
    src_blur = cv2.filter2D(src, -1, blur_filter)

    morph_filter = np.zeros((1, 50))
    morph_filter.fill(1)
    src_morph = cv2.morphologyEx(src_blur, cv2.MORPH_OPEN, morph_filter)

    morph_channel = src_blur - src_morph

    thresh_low = np.mean(morph_channel) + 1.2 * np.std(morph_channel)
    thresh_high = np.max(morph_channel)
    morph_binary = np.zeros_like(morph_channel)
    morph_binary[(morph_channel > thresh_low) &
                 (morph_channel <= thresh_high)] = 1
    return morph_binary.astype(np.uint8)


# Must improve - Include white
def environment_filter(img):
    a_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]
    a_channel = cv2.medianBlur(a_channel, 5)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hls = cv2.medianBlur(hls, 5)

    # Filter out Greenery & Soil from environment
    thresh_low = np.mean(a_channel) + 0.2 * np.std(a_channel)
    thresh_high = np.max(a_channel)
    green_binary = np.zeros_like(a_channel)
    green_binary[(a_channel >= thresh_low) & (a_channel <= thresh_high)] = 1

    return green_binary.astype(np.uint8)


def x_edge(img):
    """Finds the vertical edges - lanes lines"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take x derivative
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx = np.absolute(sobelx)
    # Convert sobelx into an 8-bit image
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Create binary image using thresholding
    sobelx_binary = np.zeros_like(scaled_sobelx)
    sobelx_binary[(scaled_sobelx > 40) & (scaled_sobelx <= np.max(scaled_sobelx))] = 255
    # Return the binary output
    return sobelx_binary.astype(np.uint8)


def image_threshold(img):
    morph_binary = morphology_filter(img)

    environment_mask = environment_filter(img).astype(np.uint8)
    x_edge_binary = x_edge(img)
    combined_binary = cv2.bitwise_and(morph_binary,
                                      environment_mask,
                                      x_edge_binary).astype(np.uint8)

    return combined_binary