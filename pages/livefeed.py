import cv2
from PIL import Image
from feature_extraction import Extractor
from gallery import Gallery
import numpy as np


def run_livefeed(g: Gallery, fe: Extractor, images, labels, features):

    path = "/home/flavio/PycharmProjects/BiometricsProject/model/haar_model/model"

    faceCascade = cv2.CascadeClassifier(path)

    video_capture = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    raw_templates, coded_labels = g.get_all_original_template(mode="coded")

    # flattening the images

    flatted_templates = [template.flatten() for template in raw_templates]

    fe.new_lda_obj()

    fe.lda_obj.fit(flatted_templates, coded_labels)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),

        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img = np.asarray(Image.fromarray(gray[y:y + h, x:x + w]).resize((60, 60), Image.ANTIALIAS))

            id = fe.lda_obj.predict(img.reshape(1, -1))

            # If confidence is less them 100 ==> "0" : perfect match
            id = g.decoding_dict[id[0]]

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
