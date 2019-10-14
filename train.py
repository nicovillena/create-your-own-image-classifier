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

def get_args():
    parser = argparse.ArgumentParser(description="Training Network")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg19'], help="Pre-trained models: vgg16 or vgg19")
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', help="Learning rate")
    parser.add_argument('--hidden_units', dest='hidden_units', default='4096', help="Number of nodes in hidden layer")
    parser.add_argument('--epochs', dest='epochs', default='5', help="Number of epochs to train the model")
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=True, help="Use GPU (True) or CPU (False) to train the model")
    return parser.parse_args()

def transform(train_dir, valid_dir, test_dir):
    train_tr = transforms.Compose([transforms.RandomRotation(20),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_tr = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    valid_tr = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_tr)
    test_data = datasets.ImageFolder(test_dir, transform = test_tr)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_tr)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    return train_loader, test_loader, valid_loader

def main():
    args = get_args()
    
    data_dir = 'ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_loader, valid_loader, test_loader = transform(train_dir, valid_dir, test_dir)
    
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == 'vgg16':
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, int(args.hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(args.hidden_units), 102),
                                 nn.LogSoftmax(dim=1))
    elif args.arch == 'vgg19':
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, int(args.hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(args.hidden_units), 102),
                                 nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = float(args.learning_rate))
    epochs = int(args.epochs)
    gpu = args.gpu
    train(epochs, train_loader, valid_loader, model, criterion, optimizer, gpu)
    
    checkpoint = {'input_size': feature_num,
              'output_size': 102,
              'model': model,
              'hidden_layers': int(args.hidden_units),
              'drop': 0.2,
              'epochs': epochs,
              'learning_rate': float(args.learning_rate),
              'batch_size': 64,
              'classifier' : classifier,
              'arch': args.arch,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

def train(epochs, train_loader, valid_loader, model, criterion, optimizer, gpu):
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    
    steps = 0
    running_loss = 0
    print_every = 64

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            if gpu and cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs.cpu()), Variable(labels.cpu())
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        if gpu and cuda:
                            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                        else:
                            inputs, labels = Variable(inputs.cpu()), Variable(labels.cpu())
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
            
if __name__ == "__main__":
    main()