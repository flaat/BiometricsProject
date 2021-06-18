import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

class Gallery:
    """
    This class is a simple images container. Here there are all the utility methods to deal with the gallery
    All the templates are memorized as dictionaries K, V where the key are the names and the values are the
    templates.
    """

    def __init__(self):

        self.img_size = 250
        self.original_template = {}
        self.coding_dict = {}
        self.decoding_dict = {}


    def get_original_template_by_name(self, name):
        if name not in self.original_template:
            raise Exception("Name Not in the gallery")
        else:
            return self.original_template[name]


    def get_all_original_template(self, mode="coded"):
        """
        :param: mode: the mode can be "coded" "hash" "string", using normal
        the names will be normal, using hash the key will the the hash of the name, using coded the
        key will be coded with numbers from 0 to n

        :return: All the template for PCA in the gallery
        """
        images = []
        labels = []

        print("Getting templates")

        if mode == "hash":
            for k, v in tqdm(self.original_template.items()):
                images += v
                labels += [hash(k)] * len(v)
        elif mode == "normal":
            for k, v in tqdm(self.original_template.items()):
                images += v
                labels += [k] * len(v)
        elif mode == "coded":
            for k, v in tqdm(self.original_template.items()):
                images += v
                labels += [self.coding_dict[k]] * len(v)


        return images, np.array(labels)

    def build_gallery(self, dataset_path: str):
        """
        :param dataset_path: The dataset path, the hierarchy has to be organized as
                            dataset
                            |----->name_1
                                    |----->photo_1
                                    |----->photo_2
                                    |----->photo_3
                            |----->name_2
                                    etc....

        :return:
        """

        path = "/home/flavio/PycharmProjects/BiometricsProject/model/haar_model/model"

        face_cascade = cv2.CascadeClassifier(path)

        directories = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

        coding_index = 0

        print("Reading images and building the gallery")

        for directory in tqdm(directories):

            key = directory

            images = [name for name in os.listdir(dataset_path + "/" + directory)]

            images_list = []

            for image in images:

                normal_image = Image.open(dataset_path + "/" + directory + "/" + image)

                normal_image = normal_image.resize((self.img_size, self.img_size), Image.ANTIALIAS).convert("L")

                np_image = np.asarray(normal_image)

                faces = face_cascade.detectMultiScale(np_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:

                    face_image = np_image[y:y + h, x:x + w]

                images_list.append(np.asarray(Image.fromarray(face_image).resize((20, 20), Image.ANTIALIAS)))

                self.original_template[key] = images_list

            self.coding_dict[key] = coding_index

            coding_index += 1

        self.decoding_dict = {v: k for k, v in self.coding_dict.items()}