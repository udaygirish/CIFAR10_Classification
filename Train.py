#!/usr/bin/env python3

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

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
from torchsummary import summary

# import Misc.ImageUtils as iu - Why this import when it is never used ?
from Network.Network import CIFAR10Model, resnet34, resnext34, densenet, mobilenetv2
from Misc.MiscUtils import *
from Misc.DataUtils import *


# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize, type="test"):
    """
    Inputs:
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch

    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels
    """
    I1Batch = []
    LabelBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet) - 1)

        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################

        I1, Label = TrainSet[RandIdx]

        # Add Random Cropping Here
        I1_rc = transforms.RandomCrop((ImageSize[0], ImageSize[1]))(I1)

        # Add Random Horizontal Flipping Here
        I1_hf = transforms.RandomHorizontalFlip(p=1)(I1)

        # Add Clahe
        I1_clahe = transforms.ToPILImage()(I1)
        I1_clahe = transforms.functional.adjust_contrast(I1_clahe, 2)
        I1_clahe = transforms.functional.adjust_brightness(I1_clahe, 0.5)
        I1_clahe = transforms.ToTensor()(I1_clahe)

        # Add Random Rotation
        I1_rr = transforms.RandomRotation(20)(I1)

        # Add Random Affine
        I1_ra = transforms.RandomAffine(30)(I1)

        # Append All Images and Mask
        if type == "test":
            no_of_aug = 0
            I1Batch.append(I1)
        else:
            no_of_aug = 1
            I1Batch.append(I1)
            I1Batch.append(I1_rc)
            # I1Batch.append(I1_hf)
            # I1Batch.append(I1_clahe)
            # I1Batch.append(I1_rr)
            # I1Batch.append(I1_ra)

        # Standardization of the batch using torch

        for i in range(no_of_aug + 1):
            LabelBatch.append(torch.tensor(Label))
    I1Batch = [(i - torch.mean(i)) / torch.std(i) for i in I1Batch]
    # Standardise the inputs
    # I1Batch = [i / 255 for i in I1Batch]  # Basic Standarisation - Divide by 255

    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    TrainLabels,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    TrainSet,
    LogsPath,
    ModelName,
):
    """
    Inputs:
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    ModelName - Name of the model to be trained
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """

    if ModelName == "custom_cnn":
        model = CIFAR10Model(InputSize=3 * 32 * 32, OutputSize=10)
    elif ModelName == "resnet34":
        model = resnet34(num_classes=10)
    elif ModelName == "resnext34":
        model = resnext34(num_classes=10)
    elif ModelName == "densenet":
        model = densenet(num_classes=10)
    elif ModelName == "mobilenetv2":
        model = mobilenetv2(num_classes=10)
    else:
        print("Model Name not found")
        raise Exception("Selected Model is not available")
        sys.exit()

    ###############################################
    # Fill your optimizer of choice here!1
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)

    # Learning rate decay Implementation
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=30, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Check if GPU is available

    if torch.cuda.is_available():
        # Set the default device to GPU
        torch.cuda.set_device(0)  # You can change 0 to the GPU index you want to use

    TestSet = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=ToTensor()
    )

    model = model.to(device)
    summary(model, (3, 32, 32))
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)
    Writer.add_graph(model, torch.rand(1, 3, 32, 32).to(device))

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        result_list_per_epoch = []
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            Batch = (Batch[0].to(device), Batch[1].to(device))

            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)
            LossThisBatch = LossThisBatch.to(device)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            # scheduler.step()

            # Changing SaveCheckPoint - TEMP
            SaveCheckPoint = 5000

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                # print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(Batch)
            result_list_per_epoch.append(result)
            # model.epoch_end(Epochs * NumIterationsPerEpoch + PerEpochCounter, result)
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.add_scalar(
                "Accuracy",
                result["acc"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )

            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
        NumTestSamples = len(TestSet)
        TestLabels = [TestSet[i][1] for i in range(NumTestSamples)]
        TestNumIterationsPerEpoch = int(NumTestSamples / MiniBatchSize)
        test_result_list_per_epoch = []
        for PerEpochCounter in tqdm(range(TestNumIterationsPerEpoch)):
            Batch = GenerateBatch(TestSet, TestLabels, ImageSize, MiniBatchSize)
            Batch = (Batch[0].to(device), Batch[1].to(device))
            result = model.validation_step(Batch)
            test_result_list_per_epoch.append(result)

        print("Training Accuracy")
        result_epoch = model.validation_epoch_end(result_list_per_epoch)
        model.epoch_end(Epochs, result_epoch)
        print("Testing Accuracy")
        test_result_epoch = model.validation_epoch_end(test_result_list_per_epoch)
        model.epoch_end(Epochs, test_result_epoch)

        # Plot Train Accuracy per epoch to tensorboar

        Writer.add_scalars(
            "Epoch Accuracy",
            {
                "train": result_epoch["acc"],
                "test": test_result_epoch["acc"],
            },
            Epochs,
        )

        Writer.add_scalars(
            "Epoch Loss",
            {
                "train": result_epoch["loss"],
                "test": test_result_epoch["loss"],
            },
            Epochs,
        )

        # If you don't flush, TensorBoard may not update until a lot of iterations!
        Writer.flush()
        #

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    # Parser Argument addition for model name

    Parser.add_argument(
        "--ModelName",
        default="custom_cnn",
        help="Model Name",
        required=False,
        choices=["custom_cnn", "resnet34", "resnext34", "densenet", "mobilenetv2"],
    )

    TrainSet = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=ToTensor()
    )
    TestSet = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=ToTensor()
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelName = Args.ModelName

    # Adding Base path  - Get Current Directory
    Basepath = str(os.getcwd())
    # Setup all needed parameters including file reading
    _, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(
        Basepath, CheckPointPath
    )

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
    print("Number of Testing Images: " + str(len(TestSet)))

    TrainOperation(
        TrainLabels,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        TrainSet,
        LogsPath,
        ModelName,
    )


if __name__ == "__main__":
    main()
