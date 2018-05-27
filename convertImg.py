import os
from PIL import Image
import sys
from imutils import paths
import random
import argparse


def change_size(path,args):
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        img=Image.open(imagePath)
        filename =os.path.basename(imagePath)
        filepath =os.path.dirname(imagePath)
        to_save =filepath +"/"+str(args.width)+filename
        new_width=args.width
        new_height=args.height
        out=img.resize((new_width,new_height),Image.ANTIALIAS)
        out.save(to_save)
        os.remove(imagePath)

parser = argparse.ArgumentParser()
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=128, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=128, help='Height of the frames in the video stream.')
args = parser.parse_args()

#python convertImg.py -wd 128 -ht 128
#change_size("data/traffic-sign224/test")
change_size("data/label/test",args)
#change_size("/home/winney/image_db/bmp")
#change_size("/home/winney/image_db/png")
