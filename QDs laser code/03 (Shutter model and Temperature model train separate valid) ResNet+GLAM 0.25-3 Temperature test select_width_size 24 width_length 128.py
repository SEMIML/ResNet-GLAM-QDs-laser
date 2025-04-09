
#coding:utf-8
# First import the package
import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import os
import math
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import pandas as pd

from datetime import datetime




def read_split_data(root: str):
    random.seed(0)  # Guaranteed reproducible randomised results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Iterate through folders, one folder corresponds to one category
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sorting to ensure consistent order across platforms
    flower_class.sort()
    # Generate category names and corresponding numeric indexes
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    #Define different folder names and their corresponding label numbers
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all image paths of the training set
    train_images_label = []  # Store the index information corresponding to the training set images
    every_class_num = []  # Store the total number of samples in each category
    # Supported file suffix types
    supported = [".NPY", ".npy"]  
    # Traverse the files in each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        #  Iterate over the paths of all files supported by supported
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sorting to ensure consistent order across platforms
        images.sort()
        # Get the index corresponding to the category
        image_class = class_indices[cla]
        # Record the number of samples in that category
        every_class_num.append(len(images))

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    
    return train_images_path, train_images_label


def read_split_data_val(root: str):
    random.seed(0)  # Guaranteed reproducible randomised results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Iterate through folders, one folder corresponds to one category
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sorting to ensure consistent order across platforms
    flower_class.sort()
    # Generate category names and corresponding numeric indexes
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    #Define different folder names and their corresponding label numbers
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_images_path = []  # Store the paths of all images in the validation set
    val_images_label = []  # Stores the index information corresponding to the validation set image    
    every_class_num = []  # Store the total number of samples in each category
    # Supported file suffix types
    supported = [".NPY", ".npy"]  
    # Traverse the files in each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # Iterate over the paths of all files supported by supported.
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sorting to ensure consistent order across platforms
        images.sort()
        # Get the index corresponding to the category
        image_class = class_indices[cla]
        # Record the number of samples in that category
        every_class_num.append(len(images))
        
        for img_path in images:

            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for val.".format(len(val_images_path)))
    assert len(val_images_path) > 0, "number of training images must greater than 0."    
    
    return val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # The Anti-Normalize Operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove the x-axis scale
            plt.yticks([])  # Remove the y-axis scale
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated losses
    accu_num = torch.zeros(1).to(device)   # The cumulative number of samples predicted correctly
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def val_evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # The cumulative number of samples predicted correctly
    accu_loss = torch.zeros(1).to(device)  # Accumulated losses

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[val epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



class MyDataSet(Dataset):
    """Custom Datasets"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)    
    
    def __getitem__(self, item):
        img = np.load(self.images_path[item])  
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img = transform(img) 
        
        img=img.transpose(0,1)
        img=img.transpose(1,2)
        
        """
        if self.transform is not None:
            img = self.transform(img)        
        """        
        
        label = self.images_class[item]
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Define the first convolutional layer, note the use of stride to reduce the length and width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If the number of channels or length and width of the input and output are inconsistent, the dimensions need to be adjusted
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        # Keep the value before the residual connection for subsequent addition
        residual = x
        
        # The first convolutional layer
        out = F.relu(self.bn1(self.conv1(x)))
        
        # The second convolutional layer
        out = self.bn2(self.conv2(out))
        
        # If the number of channels or the length and width of the input and output are inconsistent, adjust the dimensions
        if x.size(1) != out.size(1) or x.size(2) != out.size(2) or x.size(3) != out.size(3):
            residual = self.adjust_channels(residual)
        
        # Residual Connection
        out += residual
        out = F.relu(out)
        return out

class AdaptiveModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveModel, self).__init__()

        # Define the residual connection block and use stride=4 to reduce the length and width
        self.residual_block = ResidualBlock(in_channels, out_channels, stride=4)

    def forward(self, x):
        # Forward propagation of the model
        out = self.residual_block(x)
        return out

class LocalChannelAttention(nn.Module):
    def __init__(self, in_channels, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size % 2 == 1), "Kernel size must be odd"

        self.fc1 = nn.Linear(in_channels, in_channels*2)
        self.fc2 = nn.Linear(in_channels*2, in_channels*2)
        self.fc3 = nn.Linear(in_channels*2, in_channels)
        
        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape

        att = self.fc1(x.view(N, C, -1).mean(dim=2))
        att = F.relu(att)
        att = self.fc2(att)
        att = F.relu(att)
        att = self.fc3(att)
        att = F.relu(att)
        att = att.view(N, C, 1, 1)

        att = self.GAP(x).view(N, 1, C)
        att = self.conv(att)
        att = F.relu(att)
        att = att.view(N, C, 1, 1)

        return (x * att) + x

class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_2 = nn.Conv2d(int(num_reduced_channels * 4), in_channels, 1, 1)  # Adjust output channels

        self.dilated_conv3x3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 5, 1, padding=2)
        self.dilated_conv7x7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 7, 1, padding=3)

        self.batch_norm = nn.BatchNorm2d(int(num_reduced_channels * 4))

    def forward(self, feature_maps, local_channel_output):
        att = self.conv1x1_1(feature_maps)
        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        att = torch.cat((att, d1, d2, d3), dim=1)
        att = self.batch_norm(att)
        att = F.relu(att)
        att = self.conv1x1_2(att)

        att = F.interpolate(att, size=feature_maps.size()[2:], mode='bilinear', align_corners=False)

        att = att + feature_maps
        att = F.relu(att)

        return (local_channel_output * att) + local_channel_output

class GlobalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super(GlobalChannelAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape
        
        query = self.conv_q(self.GAP(x).view(N, 1, C)).relu()
        key = self.conv_k(self.GAP(x).view(N, 1, C)).relu().permute(0, 2, 1)

        query_key = torch.bmm(key, query).view(N, -1)
        query_key = query_key.softmax(-1).view(N, C, C)

        value = x.permute(0, 2, 3, 1).view(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.view(N, C, H, W)

        return x * att

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()

        self.conv1x1_q = nn.Conv2d(in_channels, num_reduced_channels, kernel_size=3, padding=1)
        self.conv1x1_k = nn.Conv2d(in_channels, num_reduced_channels, kernel_size=5, padding=2)
        self.conv1x1_v = nn.Conv2d(in_channels, num_reduced_channels, kernel_size=7, padding=3)
        self.conv1x1_att = nn.Conv2d(num_reduced_channels, in_channels, kernel_size=1)

    def forward(self, feature_maps, global_channel_output):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.reshape(N, C, -1) 
        key = self.conv1x1_k(feature_maps).reshape(N, C, -1)

        query_key = torch.bmm(key.permute(0, 2, 1), query)
        query_key = query_key.reshape(N, -1).softmax(-1)
        query_key = query_key.reshape(N, int(H * W), int(H * W))
        value = self.conv1x1_v(feature_maps).reshape(N, C, -1)
        att = torch.bmm(value, query_key).reshape(N, C, H, W)
        att = self.conv1x1_att(att)

        return (global_channel_output * att) + global_channel_output

class GLAM(nn.Module):
    def __init__(self, in_channels, num_reduced_channels, feature_map_size, kernel_size):
        super().__init__()

        self.local_channel_att = LocalChannelAttention(in_channels, feature_map_size, kernel_size)
        self.local_spatial_att = LocalSpatialAttention(in_channels, num_reduced_channels)
        self.global_channel_att = GlobalChannelAttention(feature_map_size, kernel_size)
        self.global_spatial_att = GlobalSpatialAttention(in_channels, num_reduced_channels)

        self.fusion_weights = nn.Parameter(torch.Tensor([0.333, 0.333, 0.333]))  # equal initial weights

    def forward(self, x):
        local_channel_att = self.local_channel_att(x)
        local_att = self.local_spatial_att(x, local_channel_att)
        global_channel_att = self.global_channel_att(x)
        global_att = self.global_spatial_att(x, global_channel_att)

        local_att = local_att.unsqueeze(1)
        global_att = global_att.unsqueeze(1)
        x = x.unsqueeze(1)

        all_feature_maps = torch.cat((local_att, x, global_att), dim=1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        fused_feature_maps = (all_feature_maps * weights).sum(1)

        return fused_feature_maps

# Define MLP for classification
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x

# Define the complete model
class MyModel(nn.Module):
    def __init__(self, in_channels=24, input_size=100, num_reduced_channels=8, feature_map_size=128, kernel_size=5, num_classes=3):
        super(MyModel, self).__init__()              
        
        # AdaptiveModel
        self.adaptive1 = AdaptiveModel(in_channels, in_channels*2)
        
        # GLAM
        self.glam1 = GLAM(in_channels*2, num_reduced_channels, feature_map_size//4, kernel_size)

        # AdaptiveModel
        self.adaptive2 = AdaptiveModel(in_channels*2, in_channels*4)

        # GLAM
        self.glam2 = GLAM(in_channels*4, num_reduced_channels, feature_map_size//16, kernel_size)
        
        # AdaptiveModel
        self.adaptive3 = AdaptiveModel(in_channels*4, in_channels*8)

        # GLAM
        self.glam3 = GLAM(in_channels*8, num_reduced_channels, feature_map_size//64, kernel_size)

        # MLP for classification
        self.mlp_classifier = MLPClassifier((in_channels*8)*(input_size//64)*(input_size//64), num_classes)

    def forward(self, x):
        #print("x\n",x.shape)
        # AdaptiveModel 1
        x = self.adaptive1(x)
        #print("adaptive1\n",x.shape)

        # GLAM 1
        x = self.glam1(x)
        #print("glam1\n",x.shape)
        
        # AdaptiveModel 2
        x = self.adaptive2(x)
        #print("adaptive2\n",x.shape)
        
        # GLAM 2
        x = self.glam2(x)
        #print("glam2\n",x.shape)
        
        # AdaptiveModel 3
        x = self.adaptive3(x)
        #print("adaptive3\n",x.shape)
        
        # GLAM 3
        x = self.glam3(x)
        #print("glam3\n",x.shape)
        
        # MLP for classification
        x = self.mlp_classifier(x)
        #print("mlp_classifier\n",x.shape)
        return x



# Analyze command-line parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Training loop for select_width_size.")
    parser.add_argument('--train_path', type=str, required=True, help="Path: Training data storage location")
    parser.add_argument('--val_path', type=str, required=True, help="Path: Verify data storage location")    
    parser.add_argument('--csv_path', type=str, required=True, help="Path: CSV file for saving accuracy and loss values")
    parser.add_argument('--weights_path_template', type=str, required=True, help="Path: Template for saving model weights")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of categories")
    parser.add_argument('--batch_size', type=int, default=128, help="The number of samples per batch")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training rounds")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--device', default='cuda:0', help="Device number (e.g. 0 or CPU)")
    return parser.parse_args()

def main(args):
    # Determine equipment
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    

    tb_writer = SummaryWriter()
    
    train_images_path, train_images_label = read_split_data(args.train_path)
    val_images_path, val_images_label = read_split_data_val(args.val_path)

    
    # Data transformation
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # Create training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Create validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # Load data
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)
    
    # Create an instance of MyModel
    in_channels = 24
    input_size = 128
    num_reduced_channels = 8
    feature_map_size = 128   
    kernel_size = 5

    
    ResNet_Glam_model = MyModel(in_channels, input_size, num_reduced_channels, feature_map_size, kernel_size, args.num_classes).to(device)

    
    # Set optimizer
    optimizer = optim.SGD(ResNet_Glam_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # Learning rate scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.05) + 0.05
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Ensure the directory exists
    weights_dir = args.weights_path_template
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # Training cycle
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(ResNet_Glam_model, optimizer, train_loader, device, epoch)
        scheduler.step()

        val_loss, val_acc = val_evaluate(ResNet_Glam_model, val_loader, device, epoch)

        # Record the results of training and validation
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        
        # Save Model
        # Save the model with the correct file name
        weights_path = os.path.join(weights_dir, "model-{}.pth".format(epoch))
        torch.save(ResNet_Glam_model.state_dict(), weights_path)

        # Store to CSV
        time = str("%s" % datetime.now())
        data = pd.DataFrame([[time, epoch, train_loss, train_acc, val_loss, val_acc]])
        data.to_csv(args.csv_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    args = parse_args()  # Analyze command-line parameters
    main(args)  # Pass the parsed parameters to the main function


