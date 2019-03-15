# coding=utf-8
import numpy as np
import sys
sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402
import plott

lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]

NeuNet = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                       layer, neurons_per_layer, 2)
NeuNet.read()
x = np.arange(0, len(NeuNet.fitnessHistory), 1)
y = NeuNet.fitnessHistory

plott.plott(x, y,
            "../wrt/NeuNetNorm/fitness_timeline.html",
            title="rating",
            x_label="x",
            y_label="y",
            colour="blue",
            width=3,
            mode="line")
