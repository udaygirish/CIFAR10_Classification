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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import CIFAR10Model, resnet34, resnext34, densenet, mobilenetv2
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.datasets import CIFAR10
from torchsummary import summary
import seaborn as sns
import matplotlib

matplotlib.use("TkAgg")


# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    transform = ToTensor()
    Img = transform(Img)
    # Standardize the image using torch
    Img = (Img - torch.mean(Img)) / torch.std(Img)
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")
    return cm


def TestOperation(
    ImageSize, ModelPath, TestSet, LabelsPathPred, ModelName, device_name
):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Use  CUDA Device if available
    if device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Predict output with forward pass, MiniBatchSize for Test is 1
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

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    model.to(device)
    summary(model, (3, 32, 32))
    OutSaveT = open(LabelsPathPred, "w")
    model.eval()
    t1 = time.time()
    for count in tqdm(range(len(TestSet))):
        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)
        Img = torch.from_numpy(Img)
        Img = Img.to(device)
        PredT = torch.argmax(model(Img)).item()

        OutSaveT.write(str(PredT) + "\n")
    t2 = time.time()
    OutSaveT.close()
    print("Time taken for testing: ", t2 - t1)
    print("Inference Time: ", (t2 - t1) / len(TestSet))


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="/home/aa/144model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )

    Parser.add_argument(
        "--ModelName",
        default="custom_cnn",
        help="Model Name",
        required=False,
        choices=["custom_cnn", "resnet34", "resnext34", "densenet", "mobilenetv2"],
    )

    Parser.add_argument(
        "--device",
        default="cuda",
        help="device",
        required=False,
        choices=["cuda", "cpu"],
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    ModelName = Args.ModelName
    device_name = Args.device

    TestSet = CIFAR10(root="data/", train=False)
    TrainSet = CIFAR10(root="data/", train=True)

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels

    TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelName, device_name)

    plt.figure()
    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    cm_test = ConfusionMatrix(LabelsTrue, LabelsPred)
    model_file_path_split = ModelPath.split("/")[2]
    confusion_mat_plot_test = sns.heatmap(cm_test, annot=True, fmt="d")#, cmap="Blues")
    plt.title(str(ModelName)+"_CF Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    fig_test = confusion_mat_plot_test.get_figure()
    fig_test.savefig(model_file_path_split + "_cfmatrix_test.png")

    plt.figure()
    # Running Test Operation to get Metrics on Training Set
    TestOperation(ImageSize, ModelPath, TrainSet, "./TxtFiles/PredOutTrain.txt", ModelName, device_name)
    LabelsTrue, LabelsPred = ReadLabels(
        "./TxtFiles/LabelsTrain.txt", "./TxtFiles/PredOutTrain.txt"
    )
    cm_train = ConfusionMatrix(LabelsTrue, LabelsPred)
    confusion_mat_plot_train = sns.heatmap(cm_train, annot=True, fmt="d") #, cmap="Blues")
    plt.title(str(ModelName)+"_CF Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    fig_train = confusion_mat_plot_train.get_figure()
    fig_train.savefig(model_file_path_split + "_cfmatrix_train.png")


if __name__ == "__main__":
    main()
