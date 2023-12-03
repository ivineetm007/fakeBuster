from . import FakeDetector
import keras
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

class XceptionFakeDetector(FakeDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.winlength = 1
        self.overlap = 0
        self.imgdim = 150

    def load_model(self, path):
        print(tf.config.list_physical_devices('GPU'))
        if self.device == 'cuda':
            with tf.device("gpu:0"):
                self.model = keras.models.load_model(path)
        else:
            with tf.device("cpu:0"):
                self.model = keras.models.load_model(path)


    def preprocess(self,image_list):
        '''

        :param image_list: list of RGB images
        :return: batch of images
        '''
        batch_img = []
        for image in image_list:
            im=Image.fromarray(image)
            im = im.resize((self.imgdim,self.imgdim))
            batch_img.append(np.array(im))
        return np.array(batch_img)

    def predict(self,chunkslist):
        image_list = [chunk[0] for chunk in chunkslist]
        data_batch = self.preprocess(image_list)
        data_batch = np.array(data_batch)  # shape -> [?, 150, 150, 3]
        data_batch = (data_batch - np.mean(data_batch, axis=(0, 1, 2))) / np.std(data_batch, axis=(0, 1, 2))
        fakescores = self.model.predict(data_batch)  # dim [B,1]
        return fakescores[:,0]
