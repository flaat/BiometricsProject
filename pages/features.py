import streamlit as st
from matplotlib import pyplot as plt

from feature_extraction import Extractor
from gallery import Gallery


def run_feature_extraction(g: Gallery, fe, images, labels, features):
    st.title("Features extraction")

    all_features = sorted(features.keys())
    selected_feature = st.select_slider(
        "Select one feature extractor",
        options=all_features,
    )
    idx = st.select_slider(
        "Select one index",
        options=range(len(features[selected_feature])),
    )

    if selected_feature == "gabor":
        idx_inner = st.select_slider(
            "Select inner index",
            options=range(len(features[selected_feature][idx])),
        )
        img = features[selected_feature][idx][idx_inner].reshape(60, 60, 1)
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = features[selected_feature][idx]
        img = (img - img.min()) / (img.max() - img.min())
    st.image(img, width=350)

    return
