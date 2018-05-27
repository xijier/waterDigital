# coding: utf-8

# In[1]:


# import pacakes and layer
import keras
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam

class googleNet:
    @staticmethod
    def add_module(input):
        # print(input.shape)
        Conv2D_reduce = Conv2D(16, (1, 1), strides=(2, 2), activation='relu', padding='same')(input)
        # print(Conv2D_reduce.shape)
        Conv2D_1_1 = Conv2D(16, (1, 1), activation='relu', padding='same')(input)
        # print(Conv2D_1_1.shape)
        Conv2D_3_3 = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(Conv2D_1_1)
        # print(Conv2D_3_3.shape)
        Conv2D_5_5 = Conv2D(16, (5, 5), strides=(2, 2), activation='relu', padding='same')(Conv2D_1_1)
        # print(Conv2D_5_5.shape)
        MaxPool2D_3_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input)
        # print(MaxPool2D_3_3.shape)
        Cov2D_Pool = Conv2D(16, (1, 1), activation='relu', padding='same')(MaxPool2D_3_3)
        # print(Cov2D_Pool.shape)
        concat = Concatenate(axis=-1)([Conv2D_reduce, Conv2D_3_3, Conv2D_5_5, Cov2D_Pool])
        # print(concat.shape)
        return concat
    @staticmethod
    def build(width, height, depth, classes):
        input = Input(shape=(height, width, depth,))

        Conv2D_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
        MaxPool2D_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Conv2D_1)
        BatchNorm_1 = BatchNormalization()(MaxPool2D_1)
        Module_1 = googleNet.add_module(BatchNorm_1)
        Module_1 = googleNet.add_module(Module_1)
        Output = Flatten()(Module_1)
        Output = Dense(classes, activation='softmax')(Output)
        model = Model(inputs=[input], outputs=[Output])
        # determine Loss function and Optimizer
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model