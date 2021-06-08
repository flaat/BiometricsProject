import matplotlib.pyplot as plt
import numpy as np

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """
    This function plots images

    :param images: list of images
    :param titles: list of titles (for example predicted class for each image)
    :param h: height
    :param w: width
    :param n_row: number of rows
    :param n_col: number of cols
    :return: None
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



def displayHistogram(histogram):
    """

    :param histogram: The vector with the histogram values

    :return:
    """

    axis_values = np.array([i for i in range(0, len(histogram))])
    plt.bar(axis_values, histogram)
    plt.show()
