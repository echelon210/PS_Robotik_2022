import torch
import torchvision

# This script is intended to convert the models created with higher versions of
# torch (like 1.11.0) back to the old serialization method so that they can be
# deployed on machines with lower torch versions.
# ---> For example jetbot uses torch 1.6 by default

MODEL_TYPE = 'squeezenet'
MODEL_PATH = 'CNN_models/signDetection_V2_squeeze.pth'
NUMBER_OF_OUTPUTS = 7  # 7 for sign detection or 1 for road following

# Initialize model depending on input model type
if MODEL_TYPE == 'squeezenet':
    model = torchvision.models.squeezenet1_1(pretrained=False)
    model.classifier[1] = torch.nn.Conv2d(512, NUMBER_OF_OUTPUTS, kernel_size=1)
    model.num_classes = NUMBER_OF_OUTPUTS
elif MODEL_TYPE == 'resnet':
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, NUMBER_OF_OUTPUTS)
elif MODEL_TYPE == 'alexnet':
    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, NUMBER_OF_OUTPUTS)

# Load trained model
model.load_state_dict(torch.load(MODEL_PATH))

# Save converted model
basePath = MODEL_PATH.split(".")[0]
torch.save(model.state_dict(), basePath + "_conv.pth", _use_new_zipfile_serialization=False)
