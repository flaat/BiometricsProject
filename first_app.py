import streamlit as st
import numpy as np

from feature_extraction import Extractor
from gallery import Gallery
from pages.home import run_home
from pages.livefeed import run_livefeed
from pages.features import run_feature_extraction
from pages.dataset import run_dataset


def get_pca_templates(g: Gallery, fe: Extractor, images, features):
    fe.new_pca_obj(train_set=images, n_components=150)
    eigenfaces = fe.pca_obj.components_.reshape((150, 60, 60))
    features["pca"] = eigenfaces
    return g, fe, features


def get_gabor_features(g, fe, images, features):
    fe.new_gabor_filters()

    img = images[0]
    all_gabor_features = []
    for img in images:
        gabor_features = fe.get_gabor_features(img)
        print(gabor_features)
        all_gabor_features.append(gabor_features)

    all_gabor_features = np.array(all_gabor_features)
    features["gabor"] = all_gabor_features

    return g, fe, features


@st.cache(show_spinner=False, suppress_st_warning=True)
def load_data(path):
    # build gallery
    g = Gallery()
    g.build_gallery(path)
    # loading images
    images, labels = g.get_all_original_template(mode="normal")
    flatten_images = [img.flatten() for img in images]
    images = np.array(images)

    # extract pca features
    fe = Extractor()
    features = {}

    g, fe, features = get_pca_templates(g, fe, flatten_images, features)
    g, fe, features = get_gabor_features(g, fe, flatten_images, features)

    return g, fe, images, labels, features


g, fe, images, labels, features = load_data("/home/flavio/Scrivania/dataset/lfw_funneled")


pages = {
    "Home": run_home,
    "Dataset visualization": run_dataset,
    "Feature extraction": run_feature_extraction,
    "Live feed": run_livefeed,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page(g, fe, images, labels, features)
