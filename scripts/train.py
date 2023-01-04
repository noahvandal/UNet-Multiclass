import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utilities import *
from model import UNET
from PIL import Image

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

MODEL_PATH = 'C:/Users/noahv/OneDrive/My Projects 2022 +/Ongoing/GithubPublicRepositories/UNet-Multiclass/model_full_0102.pt'
LOAD_MODEL = True
ROOT_DIR = 'C:/Users/noahv/OneDrive/My Projects 2022 +/Ongoing/GithubPublicRepositories/Datasets/Cityscapes/CITYSCAPES_DATASET'
IMG_HEIGHT = 110
IMG_WIDTH = 220
BATCH_SIZE = 16
LEARNING_RATE = 0.005
EPOCHS = 15


oldlossweights = [1, 50, 0.1, 0.01, 50,  # weights for lightly used training labels;  inference image and count pixels; give weights on following {px count, weight}
                  # {10000+:0.01,1000+:0.1,500+:0.5,100+:1,50+:5,0+10}
                  0.5, 0.1, 10, 0.01, 0.01,
                  1, 10, 1, 10, 10,
                  1, 0.1, 1, 10
                  ]

newlossweights = [0.06816045, 0.06295277, 0.14123963, 0.00031831, 0.01,
                  0.15276553, 0.10455111, 0.20506287, 0.3021719,  0.00003468,
                  0.00813295, 0.15946163, 0.11305483, 0.14339668, 0.00000267,
                  0.86129760, 0.00026585, 0.03125588, 0.04654817
                  ]

newlossweights = torch.tensor(newlossweights)
newlossweights = 1/newlossweights

lossweights = torch.tensor(newlossweights)


def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)
        # y = torch.movedim(y, 3, 1)
        # y = y.FloatTensor()

        # y = y[:, -1, :, :]
        # print(preds.shape, y.shape)

        # print(type(preds), type(y))
        # print(preds.dtype, y.dtype)
        # y = y.to(torch.float32)
        # print(preds.dtype, y.dtype)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    global epoch
    epoch = 0  # epoch is initially assigned to 0. If LOAD_MODEL is true then
    # epoch is set to the last value + 1.
    LOSS_VALS = []  # Defining a list to store loss values after every epoch

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH),
                          interpolation=Image.NEAREST),
    ])

    train_set = get_cityscapes_data(
        split='train',
        mode='fine',
        relabelled=True,
        root_dir=ROOT_DIR,
        transforms=transform,
        batch_size=BATCH_SIZE,
        numClasses=19
    )

    print('Data Loaded Successfully!')

    # Defining the model, optimizer and loss function
    unet = UNET(in_channels=3, classes=19).to(DEVICE).train()
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(weight=lossweights, ignore_index=255)

    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")

    # Training the model for every epoch.
    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e}')
        loss_val = train_function(
            train_set, unet, optimizer, loss_function, DEVICE)
        LOSS_VALS.append(loss_val)
        print(loss_val)
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': LOSS_VALS
        }, MODEL_PATH)
        print("Epoch completed and model successfully saved!")


if __name__ == '__main__':
    main()
