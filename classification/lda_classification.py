"""
Here we are going to use features extracted from PCA, LDA, LBPH, GABOR FILTERS to perform the
identification closed set task.
"""

import numpy as np
from sklearn.svm import SVC

from feature_extraction import Extractor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from gallery import Gallery
import matplotlib.pyplot as plt

GALLERY_PATH = "/home/flavio/Scrivania/dataset/lfw_funneled"

g = Gallery()

# building the gallery

g.build_gallery(GALLERY_PATH)

target_names = [v for _, v in g.decoding_dict.items()]

# getting images from the gallery

raw_templates, coded_labels = g.get_all_original_template(mode="coded")

# flattening the images

flatted_templates = [template.flatten() for template in raw_templates]

X_train, X_test, y_train, y_test = \
    train_test_split(flatted_templates, coded_labels, test_size=0.25)

fe = Extractor()

fe.new_lda_obj()

fe.lda_obj.fit(X_train, y_train)

scores = cross_val_score(fe.lda_obj, np.append(X_train, X_test, axis=0),np.append(y_train, y_test), cv=10, scoring="f1_macro")

print("Mean f1: "+str(np.mean(scores)), "Std: "+str(np.std(scores)))

y_pred = fe.lda_obj.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target_names))

plot_confusion_matrix(fe.lda_obj, X_test, y_test)

plt.title("Confusions matrix for LDA and SVM")

plt.show()
