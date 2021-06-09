from sklearn.decomposition import PCA
import cv2
import filters.gabor as gb


class Extractor:

    def __init__(self):

        self.pca_obj = None
        self.lbph_obj = None
        self.lda_obj = None
        self.gabor_filters = None


    def new_gabor_filters(self):

        self.gabor_filters = gb.build_filters()


    def get_gabor_features(self, images):

        if self.gabor_filters is None:
            print("GABOR FILTERS NOT DEFINED")
        else:
            return gb.process(images, self.gabor_filters)


    def new_lda_obj(self):

        self.lda_obj = cv2.face.FisherFaceRecognizer_create()

    def get_lda_template(self, images, labels):

        if self.lda_obj is None:
            print("LDA OBJ NOT DEFINED")
        else:
            self.lda_obj.train(images, labels)
            return self.lda_obj

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
