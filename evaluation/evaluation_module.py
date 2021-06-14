from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from gallery import Gallery
from scipy.spatial import distance
import seaborn as sns
from feature_extraction import Extractor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "/home/flavio/Scrivania/dataset/lfw_funneled"

g = Gallery()

g.build_gallery(DATASET_PATH)

# New feature extractor
fe = Extractor()

# PCA GALLERY INITIALIZATION

# In order to initialize the pca feature extractor we need to instantiate a pca object
# to do this we must get the images to train the pca obj. We get these images from
# the gallery.

train_images, labels = g.get_all_original_template(mode="coded")

target_names = [v for k, v in g.decoding_dict.items()]

flatted_images = [img.flatten() for img in train_images]


X_train, X_test, y_train, y_test = train_test_split(
    flatted_images, labels, test_size = 0.25, random_state = 42)



# fe.new_pca_obj(train_set=X_train, n_components=150)
#
# # Now we have inside the fe object the PCA obj through we can perform dimensionality reduction
# # we can obtain dimensionality reduction for the entire gallery using the code below
#
print("Computing PCA GALLERY")

# probes = fe.pca_obj.transform(X_test)
#templates = fe.pca_obj.transform(X_train)

#train_images = fe.pca_obj.transform(flatted_images)

#
# for k, v in tqdm(g.original_template.items()):
#
#     pca_templates = fe.get_pca_templates(v)
#
#     g.pca_template[k] = pca_templates
# images = []
# for img in train_images:
#
#     images.append(img.reshape(60, 60))
#
#fe.new_lbph_obj()
#
# # Get the histograms
# print(len(train_images), len(train_images[0]))
# histogram = fe.get_lbph_template(train_images, labels)
fe.new_lda_obj()
template = fe.get_lda_template(flatted_images, labels)

# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(
#     SVC(kernel ='rbf', class_weight ='balanced'), param_grid
# )
#clf = SVC(kernel ='rbf', class_weight ='balanced')
# clf.fit(X_train, y_train)
# print(clf.best_estimator_)
#
# y_pred = clf.predict(template)
#
# print(classification_report(y_test, y_pred, target_names=target_names))


def distance_matrix(template):
    """

    :param template: A list of all the template contained in the gallery
    :return:
    """
    distances = []

    total_genuine = 0
    total_impostor = 0

    for i in range(0, len(labels)):

        probe = template[i]

        temp = []

        for j in range(0, len(labels)):

            gallery_sample = template[j]

            if i != j:

                if labels[i] == labels[j]:

                    total_genuine += 1
                else:
                    total_impostor += 1

            temp.append(np.linalg.norm(probe - gallery_sample))

        distances.append(temp)

    print(len(distances), len(distances[0]))

    results = {}

    t_range = np.linspace(2, 20, 100)

    for t in tqdm(t_range):
        res = [0, 0, 0, 0]
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j:
                    if distances[i][j] < t:
                        if labels[i] == labels[j]:
                            res[0] += 1
                        else:
                            res[1] += 1
                    elif labels[i] == labels[j]:
                        res[2] += 1
                    else:
                        res[3] += 1
        results[str(t)] = [res[0] / total_genuine, res[1]/ total_impostor, res[2]/ total_genuine, res[3]/total_impostor]

    GAR = []
    FAR = []
    FRR = []

    for k, v in results.items():

        print("Threshold: "+k+" values: "+str(v))
        GAR.append(v[0])
        FAR.append(v[1])
        FRR.append(v[2])

    print("GAR: ", GAR)
    print("FRR: ", FRR)
    print("FAR: ", FAR)
    plt.plot(FAR, GAR, marker='o')
    plt.xlabel("FAR")
    plt.ylabel("GAR")
    plt.title("ROC for LDA")
    plt.show()

    #print("GENUINE ACCEPTANCE: "+str(res[0])+"\nFALSE ACCEPTANCE: "+str(res[1])+"\nFALSE REJECTION: "+str(res[2])+"\nGENUINE REJECTION: "+str(res[3]))


distance_matrix(template)
