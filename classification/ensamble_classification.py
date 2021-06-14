"""
Here we are going to use features extracted from PCA, LDA, LBPH, GABOR FILTERS to perform the
identification closed set task.
"""
import random

import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
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

for i in tqdm(range(0,10)):
    X_train_lbph, X_test_lbph, y_train, y_test = \
        train_test_split(raw_templates, coded_labels, test_size=0.25,random_state=random.randint(0,100))

    X_train = [template.flatten() for template in X_train_lbph]
    X_test = [template.flatten() for template in X_test_lbph]


    fe = Extractor()

    # LDA CLASSIFIER

    fe.new_lda_obj()

    fe.lda_obj.fit(X_train, y_train)

    # PCA CLASSIFIER

    fe.new_pca_obj(train_set=X_train, n_components=150)

    fe.pca_obj.fit(X_train)

    X_train_pca = fe.pca_obj.transform(X_train)

    X_test_pca = fe.pca_obj.transform(X_test)

    clf = SVC(kernel='rbf', class_weight='balanced')

    clf.fit(X_train_pca, y_train)

    # LBPH CLASSIFIER

    fe.new_lbph_obj()

    fe.get_lbph_template(X_train_lbph, y_train)

    y_pred = []


    # ENSEMBLE CLASSIFIER

    for pca, lbph, lda in zip(X_test_pca, X_test_lbph, X_train):

        y_pred_1, _ = fe.lbph_obj.predict(lbph)

        y_pred_2 = clf.predict(np.reshape(pca, (1, -1)))

        y_pred_3 = fe.lda_obj.predict(np.reshape(lda, (1, -1)))

        if y_pred_1 == y_pred_2 and y_pred_2 == y_pred_3:

            result = y_pred_1

        elif y_pred_1 != y_pred_2 and y_pred_2 != y_pred_3 and y_pred_1 != y_pred_3:

            result = y_pred_1  # Because it is empirically the most reliable

        elif y_pred_1 == y_pred_2:

            result = y_pred_1

        elif y_pred_2 == y_pred_3:

            result = y_pred_2[0]

        elif y_pred_1 == y_pred_3:

            result = y_pred_3[0]

        y_pred.append(result)

    scores.append(f1_score(y_test, y_pred, average='macro'))

print(classification_report(y_test, y_pred, target_names=target_names))

print("Mean f1: "+str(np.mean(scores)), "Std: "+str(np.std(scores)))
