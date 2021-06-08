from sklearn.decomposition import PCA
import cv2



class Extractor:

    def __init__(self):

        self.pca_obj = None
        self.lbph_obj = None

    def new_lbph_obj(self):

        self.lbph_obj = cv2.face.LBPHFaceRecognizer_create()

    def get_lbph_template(self, images, labels):

        if self.lbph_obj is None:
            print("LBPH OBJECT NOT DEFINED, CALL new_lbph_obj() BEFORE!")
        else:
            self.lbph_obj.train(images, labels)
            return self.lbph_obj.getHistograms()

    def new_pca_obj(self, train_set, n_components):
        """
        It defines a new PCA object

        :param train_set: Set of images to train PCA. The images are one-dimensional a list of one dimensional
                        numpy array of length height*width
        :param n_components: Number of principal components
        :return: None
        """
        print("PCA object initialization...")
        self.pca_obj = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(train_set)

    def get_pca_templates(self, images):
        # eigenfaces = self.pca_obj.components_.reshape(n_components, self.img_size, self.img_size)
        if self.pca_obj is None:
            print("PCA OBJECT NOT DEFINED, CALL new_pca_obj() BEFORE!")
        else:
            return self.pca_obj.transform(images)
