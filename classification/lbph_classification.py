"""
Here we are going to use features extracted from PCA, LDA, LBPH, GABOR FILTERS to perform the
identification closed set task.
"""
import numpy as np
from sklearn.svm import SVC
import utility as utl
from feature_extraction import Extractor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns
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

scores = []
for i in range(0, 10):

    X_train, X_test, y_train, y_test = \
        train_test_split(raw_templates, coded_labels, test_size=0.25)


    fe = Extractor()

    fe.new_lbph_obj()

    fe.get_lbph_template(X_train, y_train)

    y_pred = []

    for img in X_test:

        id, _ = fe.lbph_obj.predict(img)

        y_pred.append(id)

    scores.append(f1_score(y_test, y_pred, average='macro'))

print("Mean f1: "+str(np.mean(scores)), "Std: "+str(np.std(scores)))

X_train, X_test, y_train, y_test = \
    train_test_split(raw_templates, coded_labels, test_size=0.25)

y_pred = []

for img in X_test:
    
    id, _ = fe.lbph_obj.predict(img)

    y_pred.append(id)

print(classification_report(y_test, y_pred, target_names=target_names))

cnf_mtx = confusion_matrix(y_pred, y_test)

sns.heatmap(cnf_mtx, annot=True)

plt.title("Confusions matrix for LBPH")

plt.show()