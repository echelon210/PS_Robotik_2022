import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

""" SCRIPT TO TRAIN A CNN FOR ROAD FOLLOWING. """

# GLOBAL PARAMETERS
# Architecture
MODEL_TYPE = 'squeezenet'

# Load model or train new model
LOAD_MODEL = False
LOAD_FILE = 'CNN_models/ fill model name here .pth'

# Result model file
BEST_MODEL_PATH = 'CNN_models/signDetection_V2_squeeze.pth'

# Training parameter
DATASET_FOLDER = 'datasets/dataset_angle_4200'
NUM_EPOCHS = 1000
INIT_LEARNING_RATE = 0.001
TEST_PERCENTAGE = 0.1
BATCH_SIZE = 24
DEVICE = 'cuda'


def get_angle(path):
    """ Gets the angle value from the image filename.
        In(1): path - Path of the file
        Out(1): angle value for image from input path
    """
    # For training the squeezenet all angle values have to be shifted by 90°
    # since squeezenet can only output positive values with its standard pytorch
    # implementation. These 90° have to be subtracted again when it comes to
    # deployment of the net.
    angle = float(int(path.split("_")[1]))  # Value is second element in name string
    if MODEL_TYPE == 'squeezenet':
        angle = angle + 90.0
    return angle


class AngleDataset(torch.utils.data.Dataset):
    """ Class to define a dataset of images annoated with the angle to learn. """

    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        angle = float(get_angle(os.path.basename(image_path)))
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(
            image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([angle]).float()


if __name__ == '__main__':
    dataset = AngleDataset(DATASET_FOLDER, random_hflips=False)

    test_percent = TEST_PERCENTAGE
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - num_test, num_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    print("Loaded data with " + str(len(dataset)) + " pictures")
    print("---> " + str(len(train_dataset)) + " training pictures")
    print("---> " + str(len(test_dataset)) + " testing pictures")

    # Load pretrained architecture
    if MODEL_TYPE == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
        model.num_classes = 1
    elif MODEL_TYPE == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)

    # Load previous version of model
    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_FILE))

    device = torch.device(DEVICE)
    model = model.to(device)

    best_loss = 1e9
    optimizer = optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)  # lr=0.00001

    log = []

    # TRAINING PROCESS
    print("\nStarting Training process with " + str(NUM_EPOCHS) + " epochs")
    for epoch in range(NUM_EPOCHS):
        print("-> Epoch " + str(epoch + 1), end=' ')

        # Lower Learning rate at half of the training process
        if epoch >= int(NUM_EPOCHS * 0.5):
            optimizer.param_groups[0]['lr'] = 0.0001

        # Lower Learning rate at last 5 percent of the training process
        if epoch >= int(NUM_EPOCHS * 0.95):
            optimizer.param_groups[0]['lr'] = 0.00001

        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            test_loss += float(loss)
        test_loss /= len(test_loader)

        log.append([epoch, train_loss, test_loss])

        print('Loss_train: %f, Loss_test: %f, Learning_rate: %f, Best_loss: %f' % (
            train_loss, test_loss, optimizer.param_groups[0]['lr'], best_loss))
        if test_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss

    log = np.asarray(log)
    np.savetxt(BEST_MODEL_PATH.split(".")[0] + "_LOGGING.txt", log)
