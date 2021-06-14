import cv2
from gallery import Gallery
from feature_extraction import Extractor

# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

images, labels = g.get_all_original_template()
train_images = [img.flatten() for img in images]

fe = Extractor()

fe.new_lda_obj()
fe.get_lda_template(train_images, labels)



