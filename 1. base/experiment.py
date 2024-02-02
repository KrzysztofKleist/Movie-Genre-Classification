import torch

torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    resnet50,
    ResNet50_Weights,
    vgg16,
    VGG16_Weights,
)
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import math

from utilities import create_dir, convert_time, choose_data_params

from parse_args import parse_arguments

from movieframe_dataset import MovieFrameDataset


####################################################################################
opt = parse_arguments()

####################################################################################
DEVICE = "cuda"  # 'cuda' or 'cpu'

NUM_CLASSES = 8

BATCH_SIZE = opt[
    "batch_size"
]  # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
# the batch size, learning rate should change by the same factor to have comparable results

LR = opt["lr"]  # The initial Learning Rate
MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = opt[
    "num_epochs"
]  # Total number of training epochs (iterations over dataset)
STEP_SIZE = opt[
    "step_size"
]  # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = opt["gamma"]  # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = opt["log_frequency"]

####################################################################################
# Transforming the data
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

####################################################################################
# Creating the dataset and data loaders
train_frames_type, test_frames_type, train_list, test_list = choose_data_params(
    opt["frames"], opt["augmentation"], opt["data_distribution"]
)

train_set_all = MovieFrameDataset(
    "{}/".format(train_frames_type),
    "files/{}.tsv".format(train_list),
    transform=transform,
    raw=True,
)
test_set_all = MovieFrameDataset(
    "{}/".format(test_frames_type),
    "files/{}.tsv".format(test_list),
    transform=transform,
    raw=True,
)

train_set = torch.utils.data.Subset(
    train_set_all, list(range(0, int(0.4 * len(train_set_all))))
)
test_set = torch.utils.data.Subset(
    test_set_all, list(range(0, int(0.4 * len(test_set_all))))
)

if not opt["trial"]:
    train_set = train_set_all
    test_set = test_set_all

print("train_set:", len(train_set))
print("test_set:", len(test_set))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

####################################################################################
############ MODELS ############
if opt["model"] == "alexnet":
    ############ ALEXNET ############
    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    # Changing the number of the neurons in the last layer
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
elif opt["model"] == "resnet":
    ############ RESNET50 ############
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Changing the number of the neurons in the last layer
    model.fc = nn.Linear(2048, NUM_CLASSES)
elif opt["model"] == "vgg":
    ############ VGG16 ############
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # Changing the number of the neurons in the last layer
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

#################################
# Define loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

#################################
# By default, everything is loaded to cpu
gpumodel = model.cuda()  # this will bring the network to GPU if DEVICE is cuda

cudnn.benchmark  # Calling this optimizes runtime

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

####################################################################################
# Train the model
import time

start_time = time.time()

train_losses = []
test_losses = []

for epoch in range(NUM_EPOCHS):
    print("Starting epoch {}/{}".format(epoch + 1, NUM_EPOCHS))

    epoch_start_time = time.time()

    count_train = 0
    train_loss_sum = 0

    current_step = 0

    # Iterate over the dataset
    for images, labels in train_loader:
        # Bring data over the device of choice
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        gpumodel.train()  # Sets module in training mode

        optimizer.zero_grad()  # Zero-ing the gradients

        outputs = gpumodel(images)  # Forward pass to the network

        # Apply the model
        train_loss = criterion(outputs, labels)

        count_train += 1

        # Compute gradients for each layer and update weights
        train_loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

        train_loss_sum += train_loss.item()

        # Log loss
        if current_step % LOG_FREQUENCY == 0:
            print(
                "Step {}/{}, Loss {}, LR {}".format(
                    current_step,
                    len(train_loader),
                    train_loss_sum / count_train,
                    scheduler.get_last_lr(),
                )
            )
            train_losses.append(train_loss_sum / count_train)

            count_train = 0
            train_loss_sum = 0

            count_test = 0
            test_loss = 0

            # Run the testing batches
            for images, labels in test_loader:
                # Bring data over the device of choice
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                gpumodel.train(False)  # Sets module in testing mode

                outputs = gpumodel(images)  # Forward pass to the network

                test_loss += criterion(outputs, labels).item()

                count_test += 1

            # print('Test loss {}'.format(test_loss/count_test))

            test_losses.append(test_loss / count_test)

        current_step += 1

    print(f"Epoch time: {convert_time(time.time() - epoch_start_time)} seconds\n")
    scheduler.step()


print(
    f"\nTraining duration: {convert_time(time.time() - start_time)} seconds"
)  # print the time elapsed

####################################################################################
# Evaluate model performance
with torch.no_grad():
    gpumodel.train(False)  # Set Network to evaluation mode
    val_outputs = []
    val_losses = []

    for images, labels in test_loader:
        norm_images = []

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = gpumodel(images)
        val_outputs.append(outputs)
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())

####################################################################################
# Plot
print("train_losses: ", train_losses)
print("test_losses: ", test_losses)

create_dir("results")
create_dir(
    "results/{}_{}_{}_{}_epochs".format(
        train_frames_type, train_list, model.__class__.__name__, str(NUM_EPOCHS)
    )
)

# File path
train_losses_path = "results/{}_{}_{}_{}_epochs/{}_losses.txt".format(
    train_frames_type, train_list, model.__class__.__name__, str(NUM_EPOCHS), train_list
)
# Open the file in write mode (this will create the file if it doesn't exist)
with open(train_losses_path, "w") as file:
    # Iterate through the list and write each element to the file
    for item in train_losses:
        file.write(str(item) + "\n")

# File path
test_losses_path = "results/{}_{}_{}_{}_epochs/{}_losses.txt".format(
    train_frames_type, train_list, model.__class__.__name__, str(NUM_EPOCHS), test_list
)
# Open the file in write mode (this will create the file if it doesn't exist)
with open(test_losses_path, "w") as file:
    # Iterate through the list and write each element to the file
    for item in test_losses:
        file.write(str(item) + "\n")

# Create an X axis
num_batches_per_epoch = math.ceil(len(train_set) / BATCH_SIZE)
num_logs_per_epoch = math.ceil(num_batches_per_epoch / LOG_FREQUENCY)
interval = 1 / num_logs_per_epoch
x_axis = [i * interval for i in range(len(train_losses))]

x_int = range(math.floor(min(x_axis)), math.ceil(max(x_axis)) + 1)

plt.figure()
plt.plot(x_axis, train_losses, label="training loss")
plt.plot(x_axis, test_losses, label="validation loss")
plt.xticks(x_int)
plt.title("Loss plot")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(
    "results/{}_{}_{}_{}_epochs/{}_{}_{}_{}_losses.png".format(
        train_frames_type,
        train_list,
        model.__class__.__name__,
        str(NUM_EPOCHS),
        train_frames_type,
        train_list,
        model.__class__.__name__,
        str(NUM_EPOCHS),
    )
)


####################################################################################
def get_predictions(outputs, top_2_acceptance_rate=0.65):
    all_preds = []
    for output in outputs:
        output = torch.sigmoid(output.cpu())
        pred_list = []
        for o in output:
            idx_1 = torch.topk(o, 1)[1]
            top_1 = torch.topk(o, 1)[0].item()
            pred = (o >= o[idx_1]).type(torch.uint8)
            if torch.topk(o, 2)[0][1].item() >= top_2_acceptance_rate * top_1:
                idx_2 = torch.topk(o, 2)[1][1]
                pred = (o >= o[idx_2]).type(torch.uint8)
            pred_list.append(pred.tolist())

        all_preds = all_preds + pred_list

    return all_preds


####################################################################################
label_test = []

for images, labels in test_loader:
    label_list = []
    for single_list in labels.tolist():
        label_list.append([int(l) for l in single_list])
    label_test = label_test + label_list

label_preds = get_predictions(val_outputs)

####################################################################################
# File path
label_preds_path = "results/{}_{}_{}_{}_epochs/{}_labels.txt".format(
    train_frames_type, train_list, model.__class__.__name__, str(NUM_EPOCHS), train_list
)
# Open the file in write mode (this will create the file if it doesn't exist)
with open(label_preds_path, "w") as file:
    file.write(str(train_set_all.labels_order) + "\n")
    # Iterate through the list and write each element to the file
    for item in label_preds:
        file.write(str(item) + "\n")

# File path
label_test_path = "results/{}_{}_{}_{}_epochs/{}_labels.txt".format(
    train_frames_type, train_list, model.__class__.__name__, str(NUM_EPOCHS), test_list
)
# Open the file in write mode (this will create the file if it doesn't exist)
with open(label_test_path, "w") as file:
    file.write(str(train_set_all.labels_order) + "\n")
    # Iterate through the list and write each element to the file
    for item in label_test:
        file.write(str(item) + "\n")

####################################################################################
multilabel_confusion_matrix(label_test, label_preds)

####################################################################################
report = classification_report(
    label_test, label_preds, output_dict=True, target_names=train_set_all.labels_order
)

####################################################################################
print(
    "{:15} {:12} {:9} {:11} {}".format(
        "Genre", "Precision", "Recall", "F1-score", "Support"
    )
)
for genre, val in report.items():
    print(
        "{:15} {:>9.3f} {:>9.3f} {:>11.3f} {:>10}".format(
            genre.lower(),
            val["precision"],
            val["recall"],
            val["f1-score"],
            val["support"],
        )
    )

# File path
classification_report_path = (
    "results/{}_{}_{}_{}_epochs/classification_report.txt".format(
        train_frames_type, train_list, model.__class__.__name__, str(NUM_EPOCHS)
    )
)
# Open the file in write mode (this will create the file if it doesn't exist)
with open(classification_report_path, "w") as file:
    file.write(
        "batch_size: {}, lr: {}, num_epochs: {}, step_size: {}, gamma: {}".format(
            BATCH_SIZE, LR, NUM_EPOCHS, STEP_SIZE, GAMMA
        )
        + "\n"
    )
    file.write(
        "{:15} {:12} {:9} {:11} {}".format(
            "Genre", "Precision", "Recall", "F1-score", "Support"
        )
        + "\n"
    )
    # Iterate through the list and write each element to the file
    for genre, val in report.items():
        file.write(
            "{:15} {:>9.3f} {:>9.3f} {:>11.3f} {:>10}".format(
                genre.lower(),
                val["precision"],
                val["recall"],
                val["f1-score"],
                val["support"],
            )
            + "\n"
        )
