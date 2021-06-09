import cv2
from gallery import Gallery
from feature_extraction import Extractor
import matplotlib.pyplot as plt
import utility as utl


# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

images, labels = g.get_all_original_template()


fe = Extractor()

fe.new_gabor_filters()

img = images[0]

features = fe.get_gabor_features(img)

utl.plot_gallery(fe.gabor_filters, [" "]*12, 25, 25)
plt.show()
utl.plot_gallery(features, [" "]*12, 250, 250)
plt.show()