import matplotlib.pyplot as plt
from gallery import Gallery
import utility as utl
from math import floor
from math import sqrt

# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

images = g.get_original_template_by_name("Flavio_Giorgi")

print(images[0])
for image in images:
    plt.imshow(image)
    plt.show()

#utl.plot_gallery(images[0], "", )