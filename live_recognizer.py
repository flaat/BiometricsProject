import cv2
import matplotlib.pyplot as plt

from gallery import Gallery
import utility as utl
import numpy as np

g = Gallery()

g.build_gallery("/home/flavio/Scrivania/dataset/lfw_funneled")

path = "/home/flavio/PycharmProjects/BiometricsProject/model/haar_model/model"

faceCascade = cv2.CascadeClassifier(path)

video_capture = cv2.VideoCapture(0)

model = cv2.face.LBPHFaceRecognizer_create()

images = g.get_original_template_by_name("Flavio_Giorgi")

utl.plot_gallery(images[0:12], [" "]*12, 60,60)

plt.show()

#model.train(images, np.asarray([12]*len(images)))

model_path = "/home/flavio/PycharmProjects/BiometricsProject/model/lbph_model/model.yml"

font = cv2.FONT_HERSHEY_SIMPLEX

model.read(model_path)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = model.predict(gray[y:y + h, x:x + w])

        # If confidence is less them 100 ==> "0" : perfect match
        if confidence < 100:
            id = g.decoding_dict[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))


        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()