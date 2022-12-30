import torch
from model import UNET
from utilities import *
from utilities import save_as_images
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import tensorflow as tf
from cityscapesscripts.helpers.labels import trainId2label as t2l
from labels import name2label, id2label

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = 'cpu'
    print('Running on the CPU')

ROOT_DIR_CITYSCAPES = 'C:/Users/noahv/OneDrive/My Projects 2022 +/Ongoing/GithubPublicRepositories/Datasets/Cityscapes/CITYSCAPES_DATASET'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 600

MODEL_PATH = 'C:/Users/noahv/OneDrive/My Projects 2022 +/Ongoing/GithubPublicRepositories/UNet-Multiclass/model_full_1230.pt'

EVAL = True
PLOT_LOSS = False


def save_predictions(data, model):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data)):

            X, y, s = batch  # here 's' is the name of the file stored in the root directory
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            # print(X.shape, y.shape, predictions.shape)
            imgs = predictions
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
            pred_labels = pred_labels.float()
            # print(pred_labels.shape)

            # Remapping the labels
            pred_labels = pred_labels.to('cpu')
            pred_labels.apply_(lambda x: t2l[x].id)
            pred_labels = pred_labels.to(device)

            # Resizing predicted images too original size
            pred_labels = transforms.Resize((1024, 2048))(pred_labels)

            # imgs = plt.imshow(tf.argmax(imgs[0], axis=-1))
            # print(imgs.shape)
            imgs = onehot_to_rgb(imgs, id2label)
            print('image shape', imgs.shape)
            print(type(imgs))
            # Configure f  ilename & location to save predictions as images
            s = str(s)
            pos = s.rfind('/', 0, len(s))
            name = s[pos+1:-18]
            imgname = '/' + name + 'rgb.png'

            global location
            location = 'C:/Users/noahv/OneDrive/My Projects 2022 +/Ongoing/GithubPublicRepositories/Datasets/Cityscapes/CITYSCAPES_DATASET/output_1230'

            # save_as_images(pred_labels, location, name)

            # save_as_images(imgs, location, imgname)
            img = Image.fromarray(imgs)
            img.save(location + imgname)


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=1)
    single_layer = single_layer[0, :, :]
    print(onehot.shape, single_layer.shape)
    output = np.zeros(onehot.shape[2:4]+(3,))
    # print(color_dict.keys())
    print(output.shape)
    for k in color_dict.keys():
        # print(k)
        # print(color_dict[k])
        color = color_dict[k][7]
        # print(color)
        output[single_layer == k] = color
        print(np.sum(np.array(single_layer) == k))
        print(k, color)
        # for i_, i in enumerate(single_layer):
        # for j_, j in enumerate(i):
        # if single_layer[i_, j_] == k:
        # print(single_layer[i_, j_], color)
        # output[i_, j_] = color

    return np.uint8(output)


def evaluate(path):
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
    save_predictions(val_set, net)


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
