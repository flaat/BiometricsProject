import streamlit as st
from feature_extraction import Extractor
from gallery import Gallery


def run_home(g: Gallery, fe: Extractor, images, labels, features):
    st.title("Home")
    st.write("Use the navigation bar on the left to explore the different modules.")
    st.write("## Dataset visualization")
    st.write(
        "Use this module to check the images used to train the classifiers and from which the features have been extracted."
    )
    st.write("## Feature extraction")
    st.write(
        "Use this module to visually check the extracted features from the algorithms used."
    )
    st.write("## Live feed")
    st.write("Use this module use one classifier in real time, using the webcam.")
