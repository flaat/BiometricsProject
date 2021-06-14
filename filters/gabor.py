import numpy as np
import cv2


def build_filters(kernel_size = 5):
    filters = []

    # For different orientations
    for theta in np.arange(0, np.pi, np.pi / 4):
        # And different wavelengths of the sinusoidal factor
        for lamb in np.arange(np.pi / 4, np.pi, np.pi / 4):
            # Get a filter
            kern = cv2.getGaborKernel((kernel_size, kernel_size), 4.0, theta, lamb, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)

    return filters


    # Given an image and a set of filters, derive the response matrices

def process(img, filters):

    responses = []

    for kern in filters:

        fimg = cv2.filter2D(img, -1, kern)
        responses.append(fimg)

    return responses