from keras.applications.vgg16 import VGG16


class Vectorize:
    def __init__(self):
        self.model = VGG16(
            include_top=False, 
            weights="imagenet", 
            input_tensor=None, 
            input_shape=None, 
            pooling=None, 
            classes=128, 
            classifier_activation="softmax"
            )
