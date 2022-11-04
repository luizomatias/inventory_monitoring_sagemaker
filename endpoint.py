import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.addHandler(logging.StreamHandler(sys.stdout))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def net():
    
    logger.info("creating model net")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))
    return model


def model_fn(model_dir):
    
    logger.info("load model")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as file:
        model.load_state_dict(torch.load(file, map_location = device))

    return model


def input_fn(request_body, content_type='image/jpeg'):

    return Image.open(io.BytesIO(request_body))


def predict_fn(input_object, model):
    
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_object=test_transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda() 
    model.eval()
    with torch.no_grad():
        logger.info("predicting")
        prediction = model(input_object.unsqueeze(0))
        
    return prediction