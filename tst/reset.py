# coding=utf-8
import numpy as np
import sys
lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]

sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402
NeuNet = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                       layer, neurons_per_layer, 2)

NeuNet.reset(ask=False)
print(NeuNet.NeuNetContent)
if input("want to save it(y/n): ") == 'y':
    NeuNet.write()
