from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import utility as utl
from gallery import Gallery

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

clf = MLPClassifier(hidden_layer_sizes=(1500, 750, 350), random_state=1, max_iter=500).fit(X_train, y_train)

# scores = cross_val_score(clf, np.append(X_train, X_test, axis=0),np.append(y_train,y_test), cv=10, scoring="f1_macro")
#
# print("Mean f1: "+str(np.mean(scores)), "Std: "+str(np.std(scores)))

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target_names))

y_pred_proba = clf.predict_proba(X_test)

utl.cumulative_match_curve(y_pred_proba, y_test, "Second - Cumulative Match Characteristic curve MLP")