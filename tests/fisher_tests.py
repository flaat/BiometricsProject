import cv2
from gallery import Gallery

# New gallery object
g = Gallery()

# Build the gallery with images of enrolled subjects, the images are stored as one dimensional np arrays
g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

images, labels = g.get_all_original_template()

model = cv2.face.FisherFaceRecognizer_create()

print("training the model...")

model.train(images, labels)


