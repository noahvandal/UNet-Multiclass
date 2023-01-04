from collections import namedtuple
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
# import PIL
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# from evaluate import rgbToOnehotNew
from labels import color2label


class CityscapesDataset(Dataset):
    def __init__(self, split, relabelled, root_dir, target_type='semantic', mode='fine', transform=None, eval=False, numClasses=1):
        self.transform = transform
        self.classes = numClasses
        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        # Preparing a list of all labelTrainIds rgb and
        # ground truth images. Setting relabbelled=True is recommended.

        self.label_path = os.path.join(
            os.getcwd(), root_dir+'/'+self.mode+'/'+self.split)
        self.rgb_path = os.path.join(
            os.getcwd(), root_dir+'/leftImg8bit/'+self.split)
        city_list = os.listdir(self.label_path)
        # print('citylist', city_list)
        for city in city_list:
            temp = os.listdir(self.label_path+'/'+city)
            list_items = temp.copy()

            # 19-class label items being filtered
            for item in temp:
                if not item.endswith('gtFine_color.png', 0, len(item)):
                    list_items.remove(item)

            # defining paths
            list_items = ['/'+city+'/'+path for path in list_items]

            self.yLabel_list.extend(list_items)
            self.XImg_list.extend(
                ['/'+city+'/' +
                    path for path in os.listdir(self.rgb_path+'/'+city)]
            )

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        imgpath = self.rgb_path+self.XImg_list[index]
        # print(imgpath)
        # print(self.yLabel_list)

        ypath = self.label_path+self.yLabel_list[index]
        # print(imgpath, ypath)
        # try:
        imgpath = str(imgpath)
        ypath = str(ypath)

        print(imgpath)
        print(ypath)
        # image = Image.open(open(imgpath, 'rb'))
        print(type(imgpath))

        # testpath = 'C:/Users/noahv/OneDrive/NDSU Research/test_segformer.png'
        img = Image.open(imgpath, mode='r')
        # image = Image.Image.load(imgpath)

        # image = cv2.imread(imgpath)
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)

        print('Image loaded')
        # y = Image.open(open(ypath, 'rb'))
        y = Image.open(ypath, mode='r')
        # y = Image.Image.load(ypath)

        # y = cv2.imread(ypath)
        # y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        y = Image.fromarray(y)
        print('Mask loaded')

        if self.transform is not None:
            img = self.transform(img)
            y = self.transform(y)

        img = transforms.ToTensor()(img)

        y = np.array(y)
        y = y[:, :, 0:3]  # removing the alpha channel (not really needed)
        y = rgbToOnehotNew(y, color2label)

        y = torch.from_numpy(y)
        y = y.type(torch.LongTensor)

        if self.eval:
            return img, y, self.XImg_list[index]
        else:
            return img, y

        # except:
            # pass
        # print(imgpath)
        # print(ypath)
        # print(image.size)
        # # print(y.size)

        # if self.transform is not None:
        #     image = self.transform(image)
        #     y = self.transform(y)

        # image = transforms.ToTensor()(image)
        # y = np.array(y)
        # # print('orig y shape', y.shape)

        # y = y[:, :, 0:3]  # removing the alpha channel (not really needed)
        # # y = torch.from_numpy(y)
        # # print('yshape before', y.shape)
        # # y = y.type(torch.LongTensor)
        # # print("yshape", y.shape)

        # # y = rgbToOnehot(y, color2label, id2label, self.classes)
        # y = rgbToOnehotNew(y, color2label)
        # # print(y.shape)

        # y = torch.from_numpy(y)
        # y = y.type(torch.LongTensor)

        # if self.eval:
        #     return image, y, self.XImg_list[index]
        # else:
        #     return image, y


def rgbToOnehotNew(rgb, colorDict):
    shape = rgb.shape[:2]
    arr = np.zeros(shape, dtype=np.int16)

    W = np.power(256, [[0], [1], [2]])
    img_id = rgb.dot(W).squeeze(-1)
    values = np.unique(img_id)

    for i, c in enumerate(values):
        try:
            arr[img_id == c] = colorDict[i][7]
        except:
            pass

    return arr


# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled',  0,      255, 'void', 0, False, True, (0,  0,  0)),
    Label('ego vehicle',  1,      255, 'void', 0, False, True, (0,  0,  0)),
    Label('rectification border',  2,      255,
          'void', 0, False, True, (0,  0,  0)),
    Label('out of roi',  3,      255, 'void', 0, False, True, (0,  0,  0)),
    Label('static',  4,      255, 'void', 0, False, True, (0,  0,  0)),
    Label('dynamic',  5,      255, 'void', 0, False, True, (111, 74,  0)),
    Label('ground',  6,      255, 'void', 0, False, True, (81,  0, 81)),
    Label('road',  7,        0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk',  8,        1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking',  9,      255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10,      255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11,        2, 'construction',
          2, False, False, (70, 70, 70)),
    Label('wall', 12,        3, 'construction',
          2, False, False, (102, 102, 156)),
    Label('fence', 13,        4, 'construction',
          2, False, False, (190, 153, 153)),
    Label('guard rail', 14,      255, 'construction',
          2, False, True, (180, 165, 180)),
    Label('bridge', 15,      255, 'construction',
          2, False, True, (150, 100, 100)),
    Label('tunnel', 16,      255, 'construction',
          2, False, True, (150, 120, 90)),
    Label('pole', 17,        5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18,      255, 'object',
          3, False, True, (153, 153, 153)),
    Label('traffic light', 19,        6, 'object',
          3, False, False, (250, 170, 30)),
    Label('traffic sign', 20,        7, 'object',
          3, False, False, (220, 220,  0)),
    Label('vegetation', 21,        8, 'nature',
          4, False, False, (107, 142, 35)),
    Label('terrain', 22,        9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23,       10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24,       11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25,       12, 'human', 6, True, False, (255,  0,  0)),
    Label('car', 26,       13, 'vehicle', 7, True, False, (0,  0, 142)),
    Label('truck', 27,       14, 'vehicle', 7, True, False, (0,  0, 70)),
    Label('bus', 28,       15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29,      255, 'vehicle', 7, True, True, (0,  0, 90)),
    Label('trailer', 30,      255, 'vehicle', 7, True, True, (0,  0, 110)),
    Label('train', 31,       16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32,       17, 'vehicle', 7, True, False, (0,  0, 230)),
    Label('bicycle', 33,       18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1,       -1,
          'vehicle', 7, False, True, (0,  0, 142)),
]

name2label = {label.name: label for label in labels}


# def fullSizeTensorFromImage(img, classes):
