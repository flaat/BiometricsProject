from gallery import Gallery
from feature_extraction import Extractor
from matplotlib import pyplot as plt
import utility as utl
import numpy as np
from tqdm import tqdm

# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

# New feature extractor
fe = Extractor()

# PCA GALLERY INITIALIZATION

# In order to initialize the pca feature extractor we need to instantiate a pca object
# to do this we must get the images to train the pca obj. We get these images from
# the gallery.

train_images, _ = g.get_all_original_template(mode="normal")

fe.new_pca_obj(train_set=train_images, n_components=150)

# Now we have inside the fe object the PCA obj through we can perform dimensionality reduction
# we can obtain dimensionality reduction for the entire gallery using the code below

print("Computing PCA GALLERY")

for k, v in tqdm(g.original_template.items()):

    pca_templates = fe.get_pca_templates(v)

    g.pca_template[k] = pca_templates

# LBPH GALLERY INITIALIZATION

# In order to initialize the lbph feature extractor we need to instantiate a lbph object
# to do this we must get the images to train the lbph obj. We get these images from
# the gallery.

_, labels = g.get_all_original_template(mode="coded")

images = []

# We need to reshape the images because there where mono-dimensional arrays

print("Reshaping images")

for img in tqdm(train_images):

    images.append(img.reshape(g.img_size, g.img_size))


# build a new lbph object

fe.new_lbph_obj()

# Get the histograms

histogram = fe.get_lbph_template(images, labels)

index = 0

print("Computing LBPH GALLERY")

for label in tqdm(labels):

    g.lbph_template[g.decoding_dict[label]] = histogram[index]

    index += 1

h_1 = g.lbph_template["Colin_Powell"][0]

utl.displayHistogram(h_1)

