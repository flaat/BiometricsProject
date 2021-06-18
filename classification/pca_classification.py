"""
Here we are going to use features extracted from PCA, LDA, LBPH, GABOR FILTERS to perform the
identification closed set task.
"""
import numpy as np
from sklearn.svm import SVC
from feature_extraction import Extractor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from gallery import Gallery
import matplotlib.pyplot as plt
import utility as utl

SVM = False

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

fe.new_pca_obj(train_set=X_train, n_components=150)

fe.pca_obj.fit(X_train)

eigen = False

if eigen:

    eigenfaces = fe.pca_obj.components_.reshape((150, 60, 60))

    utl.plot_gallery(eigenfaces[0:12], [" "]*12, 60, 60)

    plt.show()

X_train_pca = fe.pca_obj.transform(X_train)

X_test_pca = fe.pca_obj.transform(X_test)



if SVM:

    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)

    #scores = cross_val_score(clf, np.append(X_train_pca, X_test_pca, axis=0),np.append(y_train,y_test), cv=10, scoring="f1_macro")

    #print("Mean f1: "+str(np.mean(scores)), "Std: "+str(np.std(scores)))

    clf.fit(X_train_pca, y_train)

    #y_pred = clf.predict(X_test_pca)

    #print(classification_report(y_test, y_pred, target_names=target_names))

    #plot_confusion_matrix(clf, X_test_pca, y_test)
    #plt.title("Confusions matrix for PCA and SVM")
    #plt.show()

    y_pred_proba = clf.predict_proba(X_test_pca)
    utl.cumulative_match_curve(y_pred_proba, y_test, "Cumulative Match Characteristic curve PCA-SVM")


else:

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train_pca, y_train)
    print(clf.get_params())
    #scores = cross_val_score(clf, np.append(X_train_pca, X_test_pca, axis=0),np.append(y_train,y_test), cv=10, scoring="f1_macro")

    #print("Mean f1: "+str(np.mean(scores)), "Std: "+str(np.std(scores)))

    y_pred = clf.predict(X_test_pca)

    print(classification_report(y_test, y_pred, target_names=target_names))

    plot_confusion_matrix(clf, X_test_pca, y_test)
    plt.title("Confusion matrix for PCA and MLP")
    plt.show()

    y_pred_proba = clf.predict_proba(X_test_pca)

    utl.cumulative_match_curve(y_pred_proba, y_test, "Cumulative Match Characteristic curve PCA-MLP")

