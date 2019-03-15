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

x = np.arange(-10, 10, 0.1, dtype=np.float32)
y = np.array([], dtype=np.float32)

for i in x:
    y = np.append(y, NeuNet.sigmoid(i))

plott.plott(x, y,
            "../wrt/NeuNetNorm/sigmoid.html",
            title="sigmoid",
            x_label="x",
            y_label="y",
            colour="blue",
            width=3,
            mode="line")
