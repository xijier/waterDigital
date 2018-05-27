from PIL import Image
import pandas as pd
from collections import namedtuple

def clipPic(filepath,filename,xmins, ymins, xmaxs, ymaxs):
    img = Image.open(filepath)
    region = (xmins, ymins, xmaxs, ymaxs)
    cropImg = img.crop(region)
    cropImg.save(r'C:\Users\15096\PycharmProjects\waterDigital\data\label\train'+'\\'+filename)
def creat_keraData(group):
    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    filenames = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'])
        xmaxs.append(row['xmax'])
        ymins.append(row['ymin'])
        ymaxs.append(row['ymax'])
        classes_text.append(row['class'].encode('utf8'))
        filenames.append(row['filename'])
        print(row['xmin'])
        print(row['ymin'])
        print(row['xmax'])
        print(row['ymax'])
        #print(r'C:\Users\15096\PycharmProjects\waterDigital\data\SelectPic\test\1'+'\\'+row['filename'])
        clipPic(r'C:\Users\15096\PycharmProjects\waterDigital\data\SelectPic\train\1'+'\\'+row['filename'],row['filename'],row['xmin'], row['ymin'], row['xmax'], row['ymax'])

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

examples = pd.read_csv('data/label/labels_train.csv')
grouped = split(examples, 'filename')
for group in grouped:
    creat_keraData(group)