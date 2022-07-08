import torch
import torchvision
import numpy as np

""" SCRIPT TO BENCHMARK CNNs REGARDING THEIR THROUGHPUT. """

model_type = 'squeezenet'
model_name = 'CNN_models/roadFollowing_V3_squeeze_conv.pth'

if model_type == 'squeezenet':
    model = torchvision.models.squeezenet1_1(pretrained=False)
    model.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
    model.num_classes = 1

if model_type == 'resnet':
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 1)

model.load_state_dict(torch.load(model_name))
device = torch.device('cuda')
model.to(device)

# Generate dummy input 
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings = np.zeros((repetitions, 1))

# Warm up the GPU
for _ in range(10):
    _ = model(dummy_input)
# Measure throughput
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("Benchmarking result for " + str(model_type))
print("Milliseconds per frame: " + str(mean_syn))
print("FPS: " + str(1000 / mean_syn))
