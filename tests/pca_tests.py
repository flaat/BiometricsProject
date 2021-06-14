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

train_images = [img.flatten() for img in train_images]

print(len(train_images), len(train_images[0]))

fe.new_pca_obj(train_set=train_images, n_components=150)

# Now we have inside the fe object the PCA obj through we can perform dimensionality reduction
# we can obtain dimensionality reduction for the entire gallery using the code below


# EXAMPLE #

# Get two template from the same subject

template_1 = g.get_original_template_by_name("Flavio_Giorgi")[0]
template_2 = g.get_original_template_by_name("Flavio_Giorgi")[1]
template_7 = g.get_original_template_by_name("Flavio_Giorgi")[7]
template_3 = g.get_original_template_by_name("Tony_Blair")[0]

pca_template_1 = fe.pca_obj.transform([template_1.flatten()])
pca_template_2 = fe.pca_obj.transform([template_2.flatten()])
pca_template_7 = fe.pca_obj.transform([template_7.flatten()])
pca_template_3 = fe.pca_obj.transform([template_3.flatten()])

print(np.linalg.norm(pca_template_2-pca_template_1))
print(np.linalg.norm(pca_template_7-pca_template_1))
print(np.linalg.norm(pca_template_1-pca_template_3))

eigenfaces = fe.pca_obj.components_.reshape((150, 60, 60))


print(len(eigenfaces))

plt.imshow(eigenfaces[0], cmap="gray")
plt.show()


