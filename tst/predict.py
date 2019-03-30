# coding=utf-8
import numpy as np
from decimal import *
import sys
sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402

lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]

NeuNet = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                       layer, neurons_per_layer, 2)
NeuNet.read()

input_layer = np.full([NeuNet.neurons_per_layer], 0.1)
print(input_layer)
output = NeuNet.predict(input_layer)
print(output)
