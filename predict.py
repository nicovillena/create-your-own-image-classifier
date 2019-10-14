import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import argparse
import json
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", dest='image_path', type=str, default='ImageClassifier/flowers/test/102/image_08012.jpg', help="Path to image")
    parser.add_argument("--checkpoint", dest='checkpoint', type=str, default='checkpoint.pth', help="Saved checkpoint")
    parser.add_argument("--top_k", dest='top_k', type=int, default=5, help="Classes to predict")
    parser.add_argument("--category_names", dest='category_names', type=str, default="ImageClassifier/cat_to_name.json", help="File containing label names")
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=True, help="Use GPU (True) or CPU (False) to train the model")
    return parser.parse_args()

def main():
    args = get_args()
    top_k = args.top_k
    gpu = args.gpu
    image_path = args.image_path
    checkpoint = args.checkpoint
    cat_names = args.category_names
    
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(checkpoint)
    model.class_to_idx = cat_to_name
    img = process_image(image_path)
    prob, label = predict(img, model, top_k, gpu, cat_to_name)
    print(prob)
    print([cat_to_name[x] for x in label])
    
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    model.load_state_dict = checkpoint['state_dict']
    optimizer = checkpoint['optimizer']
    return model
    
def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    if width == height:
        img = img.resize((256, 256))
    elif width > height:
        ratio = int(width / height)
        img = img.resize((256 * ratio, 256))
    elif width < height:
        ratio = int(height / width)
        img = img.resize((256, 256 * ratio))
        
    img = img.crop(((img.size[0]-224)/2, (img.size[1]-224)/2, (img.size[0]-224)/2 + 224, (img.size[1]-224)/2 + 224))
    img = np.array(img)/255
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img

def predict(img, model, top_k, gpu, cat_to_name):
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
 
    model.eval()        
    image = torch.from_numpy(np.array([img])).float()
    image = Variable(image)
    if cuda:
        image = image.cuda()
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    prob = torch.topk(probabilities, top_k)[0].tolist()[0]
    index = torch.topk(probabilities, top_k)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
    label = []
    for i in range(top_k):
        label.append(ind[index[i]])

    return prob, label
    
if __name__ == "__main__":
    main()