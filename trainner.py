import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys

sys.path.append('..')
from net.lenet import LeNet
from net.AlexNet import AlexNet
from net.googleNet import googleNet
from net.VGG16 import VGG16
from net.Resnet import ResNet

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 35
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 15
#norm_size = 32
#norm_size = 224
def load_data(path,args):
        print("[INFO] loading images...")
        data = []
        labels = []
        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(path)))
        random.seed(42)
        random.shuffle(imagePaths)
        # loop over the input images
        for imagePath in imagePaths:
                # load the image, pre-process it, and store it in the data list
                try:
                    image = cv2.imread(imagePath)
                    image = cv2.resize(image, (args.width, args.height))
                    image = img_to_array(image)
                    data.append(image)

                    # extract the class label from the image path and update the
                    # labels list

                    # label = int(imagePath.split(os.path.sep)[-2])
                    label =int(imagePath.split(os.path.sep)[-2].split('/')[3])
                    labels.append(label)
                except:
                    print (imagePath)

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        # convert the labels from integers to vectors
        labels = to_categorical(labels, num_classes=CLASS_NUM)
        return data ,labels

def modelChoice(args):
    print("creating model..."+args.model)
    if args.model =='googleNet':
        model = googleNet.build(width=args.width, height=args.height, depth=3, classes=CLASS_NUM)
        return model
    elif args.model =='AlexNet':
        model = AlexNet.build(width=args.width, height=args.height, depth=3, classes=CLASS_NUM)
        return model
    elif args.model =='VGG16':
       model = VGG16.build(width=args.width, height=args.height, depth=3, classes=CLASS_NUM)
       return model
    elif args.model =='ResNet':
       model = ResNet.build(width=args.width, height=args.height, depth=3, classes=CLASS_NUM)
       return model
    else:
        model = LeNet.build(width=args.width, height=args.height, depth=3, classes=CLASS_NUM)
        return model
def train(aug, trainX, trainY, testX, testY,args):
    # initialize the model
    print("[INFO] compiling model...")
    model = modelChoice(args)
    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)
    # save the model to disk
    print("[INFO] serializing network...")
    #model.save('traffic_sign.model')
    model.save('traffic_sign_'+args.model+'.model')
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #plt.savefig(args["plot"])
    plt.savefig("plot_"+args.model+".png")
    plt.show()

# python trainner.py -m googleNet
# python trainner.py -m AlexNet
# python trainner.py -m VGG16
# python trainner.py -m LetNet
# python trainner.py -m ResNet

if __name__=='__main__':
    #train_file_path = 'data/traffic-sign/train/'
    #test_file_path = 'data/traffic-sign/test/'
    parser = argparse.ArgumentParser()
    # parser.add_argument('-src', '--source', dest='video_source', type=int,
    #                     default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=32, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=32, help='Height of the frames in the video stream.')
    parser.add_argument('-m', '--model', dest='model', type=str,
                        default='LetNet', help='model choice.')
    args = parser.parse_args()
    train_file_path = 'data/label/train/'
    test_file_path = 'data/label/test/'

    trainX,trainY = load_data(train_file_path,args)
    testX,testY = load_data(test_file_path,args)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    train(aug, trainX, trainY, testX, testY,args)