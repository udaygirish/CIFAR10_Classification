"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(out, labels)
    return loss


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"loss": loss.detach(), "acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"loss": epoch_loss.item(), "acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(
                epoch, result["loss"], result["acc"]
            )
        )


class CIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        super(CIFAR10Model, self).__init__()
        # Initiate a base convolutional network with input and output size
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32)
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16)
        )
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout1 = nn.Dropout2d(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.dropout2 = nn.Dropout2d(0.3)
        self.fc2 = nn.Linear(512, OutputSize)

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        x = self.convlayer1(xb)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.dropout1(x)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        out = F.log_softmax(x, dim=1)

        return out


# RESNET Implementation


# MobileNetV2 Block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        """
        InvertedResidual module for a neural network.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value for the convolutional layers.
            expand_ratio (float): Expansion ratio for the hidden dimension.

        Attributes:
            stride (int): Stride value for the convolutional layers.
            use_res_connect (bool): Flag indicating whether to use residual connection.
            conv (nn.Sequential): Sequential container for the convolutional layers.
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the InvertedResidual module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(ImageClassificationBase):
    def __init__(self, num_classes=1000, width_multi=1.0):
        """
        Initializes the MobileNetV2 network.

        Args:
            num_classes (int): The number of output classes. Default is 1000.
            width_multi (float): Width multiplier for the network. Default is 1.0.
        """
        super(MobileNetV2, self).__init__()
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        self.last_channel = 1280

        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multi)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel

        features.append(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, bias=False)
        )
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Create MobileNetV2 model
def mobilenetv2(num_classes=10):
    """
    Creates a MobileNetV2 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        MobileNetV2: The MobileNetV2 model.
    """
    return MobileNetV2(num_classes=num_classes, width_multi=1.0)


# Resnet basic block


class ResNetBasicBlock(nn.Module):
    """
    ResNet Basic Block implementation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layers. Default is 1.

    Attributes:
        expansion (int): Expansion factor for the output channels.

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        """
        Forward pass of the ResNet Basic Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(ImageClassificationBase):
    """
    ResNet model for image classification.

    Args:
        block (nn.Module): The building block for the ResNet model.
        layers (list[int]): The number of blocks in each layer of the ResNet model.
        num_classes (int, optional): The number of output classes. Defaults to 10.
    """

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a layer of blocks in the network.

        Args:
            block (nn.Module): The block module to be used in the layer.
            out_channels (int): The number of output channels for each block.
            blocks (int): The number of blocks to be created in the layer.
            stride (int, optional): The stride value for the first block. Defaults to 1.

        Returns:
            nn.Sequential: The layer of blocks.
        """
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Create ResNet-34 model
def resnet34(num_classes=10):
    """
    Create a ResNet-34 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ResNet: ResNet-34 model.
    """
    return ResNet(ResNetBasicBlock, [3, 4, 6, 3], num_classes)


# ResNext Block


class ResNeXtBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, cardinality, stride=1):
        """
        Initializes a ResNeXtBlock object.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            cardinality (int): Number of groups in the 1x1 convolutional layer.
            stride (int, optional): Stride value for the 3x3 convolutional layer. Defaults to 1.
        """
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        """
        Performs forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNeXt(ImageClassificationBase):
    def __init__(self, block, layers, cardinality, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cardinality, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, cardinality, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, cardinality, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnext34(num_classes=10, cardinality=32):
    return ResNeXt(ResNeXtBlock, [3, 4, 6, 3], cardinality, num_classes)


# Understand more around the cardinality concept

# Dense Net - Base 121 Deep Layer Implementation


class DenseNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        """
        Initializes a basic block for DenseNet.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            dropRate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(DenseNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        """
        Forward pass of the DenseNet basic block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        """
        BottleneckBlock is a building block for a bottleneck residual block in a neural network.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            dropRate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(
            inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        """
        Forward pass of the BottleneckBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        """
        Initializes a TransitionBlock object.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            dropRate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        """
        Forward pass of the TransitionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        """
        DenseBlock class represents a dense block in a DenseNet architecture.

        Args:
            nb_layers (int): Number of layers in the dense block.
            in_planes (int): Number of input channels.
            growth_rate (int): Number of output channels for each layer in the dense block.
            block (nn.Module): The block module to be used in the dense block.
            dropRate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, growth_rate, nb_layers, dropRate
        )

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        """
        Helper method to create the layers in the dense block.

        Args:
            block (nn.Module): The block module to be used in the dense block.
            in_planes (int): Number of input channels.
            growth_rate (int): Number of output channels for each layer in the dense block.
            nb_layers (int): Number of layers in the dense block.
            dropRate (float): Dropout rate.

        Returns:
            nn.Sequential: Sequential container of layers in the dense block.
        """
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the dense block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layer(x)


class DenseNet121(ImageClassificationBase):
    def __init__(
        self,
        depth,
        num_classes,
        growth_rate=12,
        reduction=0.5,
        bottleneck=True,
        dropRate=0.0,
    ):
        """
        Initializes a DenseNet121 network.

        Args:
            depth (int): The depth of the network.
            num_classes (int): The number of output classes.
            growth_rate (int, optional): The growth rate of the network. Defaults to 12.
            reduction (float, optional): The reduction factor for transition blocks. Defaults to 0.5.
            bottleneck (bool, optional): Whether to use bottleneck blocks. Defaults to True.
            dropRate (float, optional): The dropout rate. Defaults to 0.0.
        """
        super(DenseNet121, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = DenseNetBasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(
            in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate
        )
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(
            in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate
        )
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        """
        Performs forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


def densenet(num_classes=10):
    """
    Creates a DenseNet model for image classification.

    Parameters:
    - num_classes (int): The number of classes for classification. Default is 10.

    Returns:
    - model: The DenseNet model.
    """
    model = DenseNet121(
        num_classes=num_classes,
        depth=121,
        growth_rate=32,
        bottleneck=True,
        dropRate=0.2,
    )
    return model
