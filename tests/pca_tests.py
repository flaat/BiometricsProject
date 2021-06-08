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

# In order to initialize the pca feature extractor we need to instantiate a pca object
# to do this we must get the images to train the pca obj. We get these images from
# the gallery.

train_images, _ = g.get_all_original_template(mode="normal")

fe.new_pca_obj(train_set=train_images, n_components=150)

# Now we have inside the fe object the PCA obj through we can perform dimensionality reduction
# we can obtain dimensionality reduction for the entire gallery using the code below

for k, v in g.original_template.items():

    pca_templates = fe.get_pca_templates(v)

    g.pca_template[k] = pca_templates

# Now we are going to plot images with the respective distance from a given image

# This subject will be the claimed identity

claimed = "Jennifer_Capriati"

# This subject will be the impostor

impostor = "Jeb_Bush"

impostor_acquisition = g.get_original_template_by_name(impostor)[0]

impostor_features = fe.get_pca_templates(np.expand_dims(impostor_acquisition, 0))

templates = g.get_original_template_by_name(claimed)

pca_templates = g.get_pca_template_by_name(claimed)

distances = ["Reference"]

for template in pca_templates[0:9]:

    distances.append(str(np.linalg.norm(impostor_features - template)))

utl.plot_gallery(images=[impostor_acquisition] + templates[0:9], titles=distances, h=g.img_size, w=g.img_size, n_row=1, n_col=9)

distances = ["Reference"]

for template in g.get_pca_template_by_name(impostor)[1:10]:

    distances.append(str(np.linalg.norm(impostor_features - template)))

utl.plot_gallery(images=g.get_original_template_by_name(impostor)[0:10], titles=distances, h=g.img_size, w=g.img_size, n_row=1, n_col=9)
plt.show()
