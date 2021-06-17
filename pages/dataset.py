import numpy as np
import streamlit as st

from feature_extraction import Extractor
from gallery import Gallery


@st.cache(show_spinner=False)
def images_by_labels(images, labels):
    labels_to_imgs = {}
    unique_labels = np.unique(labels)
    for idx in unique_labels:
        labels_to_imgs[idx] = images[labels == idx]
    return labels_to_imgs


def run_dataset(g: Gallery, f1: Extractor, images, labels, features):
    st.title("Dataset visualization")
    labels_to_imgs = images_by_labels(images, labels)

    identities = sorted(labels_to_imgs.keys())
    identity = st.select_slider(
        "Select an identity",
        options=identities,
    )
    idx = st.select_slider(
        "Select one image",
        options=range(len(labels_to_imgs[identity])),
    )
    st.image(labels_to_imgs[identity][idx], width=350)

    return
