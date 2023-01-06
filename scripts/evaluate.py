import torch
from model import UNET
from utilities import *
# from utilities import save_as_images
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import tensorflow as tf
from cityscapesscripts.helpers.labels import trainId2label as t2l
from labels import name2label, id2label, color2label
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = 'cpu'
    print('Running on the CPU')

ROOT_DIR_CITYSCAPES = 'C:/Users/noahv/OneDrive/MyProjects2022Onward/CITYSCAPES_DATASET'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 600

MODEL_PATH = 'C:/Users/noahv/OneDrive/MyProjects2022Onward/Ongoing/GithubPublicRepositories/UNet-Multiclass/model_full_0102.pt'

EVAL = True
PLOT_LOSS = False


def save_predictions(data, model, globalSum):
    model.eval()
    with torch.no_grad():
        globalSum = np.zeros([19])
        for idx, batch in enumerate(tqdm(data)):

            # here 's' is the name of the file stored in the root directory
            X, y, s, weights = batch
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            # print(X.shape, y.shape, predictions.shape)
            # print(predictions)
            imgs = predictions
            # print(imgs.shape)
            # predictions = torch.nn.functional.softmax(predictions, dim=1)
            # print(predictions.shape)
            # print(predictions)
            # predictions = torch.argmax(predictions, dim=1)
            # print(predictions.shape)
            # print(predictions)
            # print(predictions[0][0][0])
            # print(imgs[0][:][0][0])
            # pred_labels = pred_labels.float()
            # print(pred_labels.shape)

            # Remapping the labels
            # pred_labels = pred_labels.to('cpu')
            # pred_labels.apply_(lambda x: t2l[x].id)
            # pred_labels = pred_labels.to(device)

            # Resizing predicted images too original size
            # pred_labels = transforms.Resize((1024, 2048))(pred_labels)

            # imgs = plt.imshow(tf.argmax(imgs[0], axis=-1))
            # print(imgs.shape)
            # imgs, globalSum = onehot_to_rgb(imgs, id2label, globalSum)
            imgs, globalSum = onehot_to_rgb(imgs, color2label, globalSum)

            # print('image shape', imgs.shape)
            # print(type(imgs))
            # Configure f  ilename & location to save predictions as images
            s = str(s)
            pos = s.rfind('/', 0, len(s))
            name = s[pos+1:-18]
            imgname = '/' + name + '0105rgb.png'

            global location
            location = 'C:/Users/noahv/OneDrive/MyProjects2022Onward/CITYSCAPES_DATASET/output_0104'

            # save_as_images(pred_labels, location, name)

            # save_as_images(imgs, location, imgname)
            img = Image.fromarray(imgs)
            img.save(location + imgname)

            print(globalSum)


# def onehot_to_rgb(onehot, color_dict, globalSum):
#     print(onehot.shape)
#     single_layer = np.argmax(onehot, axis=1)
#     single_layer = single_layer[0, :, :]
#     output = np.zeros(single_layer.shape[:2]+(3,))
#     print(single_layer.shape, output.shape)
#     print(len(color_dict.keys()))
#     for k in color_dict.keys():
#         color = color_dict[k][7]
#         output[single_layer == k] = color
#         pixelSum = np.sum(np.array(single_layer) == k)
#         globalSum[k] += pixelSum

#     return np.uint8(output), globalSum

# def onehot_to_rgb(onehot, colorDict):
#     # print(onehot.shape)
#     # single_layer = np.argmax(onehot, axis=1)
#     # single_layer = single_layer[0, :, :]
#     # output = np.zeros(single_layer.shape[:2]+(3,))
#     # print(single_layer.shape, output.shape)
#     # print(len(color_dict.keys()))
#     # for i, k in enumerate(color_dict.keys()):
#     #     color = k
#     #     output[single_layer == i] = color
#     #     pixelSum = np.sum(np.array(single_layer) == k)
#     #     globalSum[k] += pixelSum
#     print(type(onehot))
#     print(onehot.shape)
#     singleLayer = np.argmax(onehot, axis=1)
#     singleLayer = np.transpose(singleLayer, [2, 1, 0])
#     print(singleLayer)

#     dense = np.zeros(singleLayer.shape[0:2]+(3,))
#     print(singleLayer.shape, dense.shape)
#     for label, color in enumerate(colorDict.keys()):
#         if label < 19:
#             print(label, color)
#             print(singleLayer.shape, dense.shape)
#             dense[np.all(singleLayer == label, axis=-1)] = color

#     return np.uint8(dense)

def onehot_to_rgb(onehot, color_dict, globalSum):
    onehot = np.array(onehot)
    single_layer = np.argmax(onehot, axis=1)
    output = np.zeros(onehot.shape[2:4]+(3,))
    single_layer = np.transpose(single_layer, [1, 2, 0])
    # single_layer = np.concatenate(
    # (single_layer, single_layer, single_layer), axis=2)
    for i, k in enumerate(color_dict.keys()):
        # print(single_layer.shape, output.shape, i, k)
        # output[single_layer == i] = k
        if i < 19:
            output[np.all(single_layer == i, axis=-1)] = k
            pixelSum = np.sum(np.array(single_layer) == k)
            globalSum[i] += pixelSum
    return np.uint8(output), globalSum


def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) ==
                              color_dict[i], axis=1).reshape(shape[:2])

    return arr


# input rgb as numpy array; numClasses must be specified in case dictionary is not desired to be entirely trained on
def rgbToOnehot(rgb, colorDict, idDict, numClasses):
    shape = rgb.shape[:2]
    arr = np.zeros(shape, dtype=np.int16)
    # print(colorDict.keys())
    # print(idDict.keys())
    # sorting through each color in dictionary per class
    for k, j in zip(colorDict.keys(), idDict.keys()):
        if j <= (numClasses - 1):
            color = np.array(colorDict[k][7])
            for _x, x in enumerate(rgb):
                for _y, y in enumerate(x):
                    # print(color, rgb[_x][_y])
                    pixel = np.array(rgb[_x][_y])
                    if np.all(pixel == color):
                        arr[_x][_y] = j  # assigning pixel to ID value
    return arr

# input rgb as numpy array; numClasses must be specified in case dictionary is not desired to be entirely trained on


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


def evaluate(path):
    globalSum = np.zeros(34)
    T = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH),
                          interpolation=Image.NEAREST)
    ])

    val_set = get_cityscapes_data(
        root_dir=ROOT_DIR_CITYSCAPES,
        split='val',
        mode='fine',
        relabelled=True,
        transforms=T,
        shuffle=True,
        eval=True
    )

    print('Data has been loaded!')

    net = UNET(in_channels=3, classes=19).to(device)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'{path} has been loaded and initialized')
    save_predictions(val_set, net, globalSum)
    # print(globalSum)


def plot_losses(path):
    checkpoint = torch.load(path)
    losses = checkpoint['loss_values']
    epoch = checkpoint['epoch']
    epoch_list = list(range(epoch))

    plt.plot(epoch_list, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss over {epoch+1} epoch/s")
    plt.show()


if __name__ == '__main__':
    if EVAL:
        evaluate(MODEL_PATH)
    if PLOT_LOSS:
        plot_losses(MODEL_PATH)
