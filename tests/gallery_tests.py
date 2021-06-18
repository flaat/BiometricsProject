import random

import matplotlib.pyplot as plt
from gallery import Gallery
import utility as utl
from math import floor
from math import sqrt

# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

images, labels = g.get_all_original_template(mode="normal")

img_list = []
lbl_list = []

for i in range(0,12):
    k = random.randint(0, len(images))
    img_list.append(images[k])
    lbl_list.append(labels[k])


utl.plot_gallery(img_list, lbl_list, 60, 60 )
plt.show()

plt.imshow(g.get_original_template_by_name("Tony_Blair")[0], cmap="gray")
plt.show()