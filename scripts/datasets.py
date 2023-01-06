from collections import namedtuple
# import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import time
# import PIL
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# from evaluate import rgbToOnehotNew
from labels import color2label, id2label


class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False, numClasses=1):
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

        # self.label_path = os.path.join(
        #     root_dir + '/' + self.mode + '/' + self.split + '/bremen/')
        # self.rgb_path = os.path.join(
        #     root_dir + '/leftImg8bit/' + self.split + '/bremen/')

        # temp = os.listdir(self.label_path)
        # list_items = temp.copy()
        # for item in temp:
        #     if not item.endswith('gtFine_color.png', 0, len(item)):
        #         list_items.remove(item)

        # self.yLabel_list.extend(list_items)
        # tempXpath = os.listdir(self.rgb_path)
        # self.XImg_list.extend(tempXpath)

        # # print(len(self.yLabel_list), len(self.XImg_list))

        # imgItems = os.listdir(self.rgb_path)
        # maskItems = os.listdir(self.label_path)

        # list_items = maskItems.copy()
        # print(len(list_items))
        # for item in maskItems:
        #     if not item.endswith('gtFine_color.png', 0, len(item)):
        #         list_items.remove(item)

        # # print(imgItems)

        # maskItems = [self.label_path + path for path in list_items]
        # imgItems = [self.rgb_path + path for path in imgItems]

        # self.yLabel_list.extend(maskItems)
        # self.XImg_list.extend(imgItems)

        # print(len(maskItems), len(imgItems))

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
        # time.sleep(1)

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        # globalSum = []
        # print(len(self.XImg_list))
        # print(len(self.yLabel_list))
        # print(index)
        # if (index >= (len(self.XImg_list)) or (index >= len(self.yLabel_list))):
        # index = int(len(self.XImg_list) / 2)
        # print(len(self.XImg_list))
        # print(len(self.yLabel_list))
        # print(index)
        # print(self.rgb_path+self.XImg_list[index])
        # print(self.label_path+self.yLabel_list[index])

        # print(self.XImg_list)
        image = Image.open(os.path.join(self.rgb_path+self.XImg_list[index]))
        # image = open(self.rgb_path+self.XImg_list[index])
        # fd = image.fileno()
        # print(fd)
        y = Image.open(os.path.join(self.label_path+self.yLabel_list[index]))
        # y = open(self.label_path+self.yLabel_list[index])

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image)
        # y = np.array(y)
        # y = torch.from_numpy(y)

        y = np.array(y)
        # print('yshape ', y.shape)
        y = y[:, :, 0:3]  # removing the alpha channel (not really needed)
        # print('yshape', y.shape)
        y, sum = DatasetrgbToOnehotNew(
            y, color2label, id2label, self.classes)
        # print(y)

        y = torch.from_numpy(y)
        y = y.type(torch.LongTensor)

        # y = y.type(torch.LongTensor)
        # print(self.rgb_path+self.XImg_list[index])
        if self.eval:
            return image, y, self.XImg_list[index], sum
        else:
            return image, y, sum


def DatasetrgbToOnehotNew2(rgb_arr, color_dict, iddict, numclasses):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) ==
                              color_dict[i], axis=1).reshape(shape[:2])
    return arr


def DatasetrgbToOnehotNew(rgb, colorDict, iddict, numclass):
    # arr = np.zeros(rgb.shape[:2])

    # for i, clr in enumerate(colorDict.keys()):
    #     for _x, x in enumerate(rgb):
    #         for _y, y in enumerate(x):
    #             pixel = np.array(rgb[_x][_y])
    #             if np.all(pixel == clr):
    #                 arr[_x][_y] = i
    # globalSum = []
    sum = []
    dense = np.zeros(rgb.shape[:2])
    for label, color in enumerate(colorDict.keys()):
        if label < 19:
            dense[np.all(rgb == color, axis=-1)] = label
            pixelSum = np.sum(np.array(rgb) == color)
            if pixelSum == 0:
                pixelSum = 1
            sum.append(pixelSum)

    # print(dense)
    return dense, sum


def DatasetrgbToOnehotNewTest(rgb, colorDict, idDict, numClasses):
    print(rgb.shape)

    # shape = rgb[]
    # shape = np.array(rgb[:2])
    # print('shape of shape', shape.shape)
    # arr = np.zeros(shape, dtype=np.int16)
    arr = np.zeros(rgb.shape[:2])
    # W = np.power(256, [[0], [1], [2]])
    # img_id = rgb.dot(W).squeeze(-1)
    # values = np.unique(img_id)
    # print(len(values))

    # print(len(colorDict.keys()))

    # for i, c in enumerate(colorDict.keys()):
    #     print(c)
    #     print(img_id)
    #     try:
    #         arr[img_id == c] = colorDict[i][7]
    #     except:
    #         pass
    globalSum = []
    # for k, j in zip(colorDict.keys(), idDict.keys()):
    #     if j <= (numClasses - 1):
    #         color = np.array(colorDict[k][7])
    #         for _x, x in enumerate(rgb):
    #             for _y, y in enumerate(x):
    #                 # print(color, rgb[_x][_y])
    #                 pixel = np.array(rgb[_x][_y])
    #                 if np.all(pixel == color):
    #                     arr[_x][_y] = j  # assigning pixel to ID value
    #     pixelSum = np.sum(np.array(single_layer) == k)
    #     globalSum[k] += pixelSum
    # return arr
    k = 0
    # for inst in colorDict:
    # print(inst)
    # for inst in idDict:
    # print(inst)
    # print(idDict)
    for i, k in enumerate(colorDict.keys()):
        print('k', k)
        # color = colorDict[k][7]
        color = k
        ID = idDict[i][1]
        print('id', ID)
        print('arr shpae', arr.shape)
        arr[rgb == color] = ID
        pixelSum = np.sum(np.array(rgb) == color)
        globalSum.append(pixelSum)
    print(globalSum)
    arr = arr[1]
    print(arr.shape)
    # print(arr.shape)
    # print(arr[0])
    # print(arr[1])
    # print(arr[2])

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
