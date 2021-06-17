import streamlit as st
import cv2

from gallery import Gallery


def run_livefeed(g: Gallery, fe, images, labels):
    st.title("Live feed")
    run = st.checkbox("Run now")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the output
        FRAME_WINDOW.image(frame)
    else:
        pass
        # st.write('Stopped')
