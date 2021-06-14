from gallery import Gallery
from feature_extraction import Extractor
from matplotlib import pyplot as plt
from PIL import Image
import utility as utl
import numpy as np

# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

# New feature extractor
fe = Extractor()

# In order to initialize the lbph feature extractor we need to instantiate a lbph object
# to do this we must get the images to train the lbph obj. We get these images from
# the gallery.

train_images, labels = g.get_all_original_template(mode="coded")

images = []

# We need to reshape the images because there where mono-dimensional arrays

# for img in train_images:
#
#     images.append(img.reshape(g.img_size, g.img_size))

fe.new_lbph_obj()

histograms = fe.get_lbph_template(train_images, labels)

fe.lbph_obj.save("/home/flavio/PycharmProjects/BiometricsProject/model/lbph_model/model.yml")