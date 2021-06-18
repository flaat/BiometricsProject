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



def displayHistogram(histogram, title):
    """

    :param histogram: The vector with the histogram values

    :return:
    """

    axis_values = np.array([i for i in range(0, len(histogram))])
    plt.bar(axis_values, histogram)
    plt.title(title)
    plt.show()


def cumulative_match_curve(y_pred_proba, y_test, title):

    CMS = []

    for k in range(1, 19):

        numerator = 0

        for real, dist in zip(y_test, y_pred_proba):

            for i in range(0, k):

                if real == np.argmax(dist):

                    numerator += 1
                    break

                else:

                    dist[np.argmax(dist)] = 0

        CMS.append((numerator / len(y_test)))

    plt.plot(np.linspace(1, 18, 18), CMS)
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.xticks(np.arange(1, 19, step=1))
    plt.show()
