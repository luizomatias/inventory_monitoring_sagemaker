import numpy as np
import torchvision
import argparse
import logging
import os
import sys
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):

    logger.info("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects / len(test_loader)

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer, device):

    epochs = 10
    best_loss = 1e6
    image_dataset = {"train": train_loader, "valid": validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for phase in ["train", "valid"]:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])

            if phase == "valid":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

            logger.info(
                "Accuracy: {:.2f}, Loss: {:.2f}, Best loss {:.2f}".format(
                    epoch_acc, epoch_loss, best_loss
                )
            )

        if loss_counter == 1:
            break

    return model


def net():

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 5)
    )
    return model


def create_data_loaders(data, batch_size):

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(data, transform=train_transform)  # dataset

    total = len(dataset)
    train_length = int(np.ceil(0.6 * total))
    test_length = int(np.floor(0.2 * total))
    lengths = [train_length, test_length, test_length]

    trainset, testset, validset = torch.utils.data.random_split(dataset, lengths)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, validloader


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")

    logger.info(f"Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}")
    logger.info(f"Data Paths: {args.data_dir}")

    logger.info("Initializing the model.")
    model = net()
    model = model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    logger.info("Loading data")
    train_loader, test_loader, valid_loader = create_data_loaders(
        args.data_dir, args.batch_size
    )

    logger.info("Training the model.")
    model = train(model, train_loader, valid_loader, loss_criterion, optimizer, device)

    logger.info("Testing the model.")
    test(model, test_loader, loss_criterion, device)

    logger.info("Saving the model.")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )

    # Container environment
    parser.add_argument(
        "--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    args = parser.parse_args()

    main(args)
