import torch
import sys
import torch.nn as nn
from Network.Network import CIFAR10Model, resnet34,resnext34, densenet, mobilenetv2
from torchview import draw_graph

# Don't generate pyc codes
sys.dont_write_bytecode = True

model_graph = draw_graph(
    mobilenetv2(), input_size=(1,3,32,32),
    graph_name='MobileNetV2',
    roll=True,
    save_graph=True
)

# model_graph.visual_graph()