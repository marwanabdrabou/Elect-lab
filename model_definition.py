import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Cropping2D, Dropout
from tensorflow import keras


class SegmentationModel(): 
    def __init__(self): 
        inply = Input((128, 128, 1))
        # contraction path "Encoder"
        conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(inply)
        conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv1)
        pool1 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv1)
        drop1 = Dropout(0.2)(pool1)

        conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(drop1)
        conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv2)
        pool2 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv2)
        drop2 = Dropout(0.2)(pool2)

        conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(drop2)
        conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(conv3)
        pool3 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv3)
        drop3 = Dropout(0.2)(pool3)

        conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(drop3)
        conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(conv4)
        pool4 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv4)
        drop4 = Dropout(0.2)(pool4)

        convm = Conv2D(1024, (3,3), activation = 'relu', padding = 'same')(drop4)
        convm = Conv2D(1024, (3,3), activation = 'relu', padding = 'same')(convm)

        tran5 = Conv2DTranspose(512, (2,2), strides = 2, padding = 'valid', activation = 'relu')(convm)
        conc5 = Concatenate()([tran5, conv4])
        conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(conc5)
        conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(conv5)
        drop5 = Dropout(0.1)(conv5)

        tran6 = Conv2DTranspose(256, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop5)
        conc6 = Concatenate()([tran6, conv3])
        conv6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(conc6)
        conv6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(conv6)
        drop6 = Dropout(0.1)(conv6)

        tran7 = Conv2DTranspose(128, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop6)
        conc7 = Concatenate()([tran7, conv2])
        conv7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conc7)
        conv7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv7)
        drop7 = Dropout(0.1)(conv7)

        tran8 = Conv2DTranspose(64, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop7)
        conc8 = Concatenate()([tran8, conv1])
        conv8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conc8)
        conv8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv8)
        drop8 = Dropout(0.1)(conv8)

        u9 = Conv2D(1, (1,1), activation = 'relu', padding = 'same')(drop8)
        self.model = Model(inputs=inply, outputs=u9)
