import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

### GLOBAL PARAMETERS ###
# Architecture
MODEL_TYPE = 'squeezenet'

# Load model or train new model
LOAD_MODEL = False
LOAD_FILE = ' .pth'

# Result model file
BEST_MODEL_PATH = 'CNN_models/signDetection_V2_squeeze.pth'

# Training parameter
DATASET_FOLDER = 'dataset_4_masked'
NUM_EPOCHS = 200
INIT_LEARNING_RATE = 0.0001
TEST_PERCENTAGE = 0.15
BATCH_SIZE = 16
DEVICE = 'cuda'

if __name__ == '__main__':
    dataset = datasets.ImageFolder(
        DATASET_FOLDER,
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    test_size = int(len(dataset) * TEST_PERCENTAGE)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    print("Dataset has: " + str(len(dataset)) + " images in total")
    print("--> Testing dataset: " + str(len(test_dataset)))
    print("--> Training dataset: " + str(len(train_dataset)))

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

    # Load pretrained architecture
    if MODEL_TYPE == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 7)
    elif MODEL_TYPE == "squeezenet":
        model = torchvision.models.squeezenet1_1(pretrained = True)
        model.classifier[1] = torch.nn.Conv2d(512, 7, kernel_size = 1)
        model.num_classes = 7
    
    # Load previous version of model
    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_FILE))

    device = torch.device(DEVICE)
    model = model.to(device)

    best_accuracy = 0.0
    optimizer = optim.SGD(model.parameters(), lr=INIT_LEARNING_RATE, momentum=0.6)

    log = []

    # TRAINING PROCESS
    print("\nSTARTING TRAINING PROCESS: ")
    for epoch in range(NUM_EPOCHS):

        # Lower Learning rate at last 5 percent of the training process
        if epoch >= int(NUM_EPOCHS * 0.95):
            optimizer.param_groups[0]['lr'] = 0.00001
        
        train_error_count = 0.0
        for images, labels in iter(train_loader):
            labels = labels
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            labels = labels
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        
        train_accuracy = 1.0 - float(train_error_count) / float(len(train_dataset))
        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))

        log.append([epoch, train_accuracy, test_accuracy])

        print('-> Epoch %d:   Accuracy Train = %f   Accuracy Test = %f' % ((epoch + 1), train_accuracy, test_accuracy))
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy

        np.savetxt(BEST_MODEL_PATH.split(".")[0] + "_LOGGING.txt", np.asarray(log))