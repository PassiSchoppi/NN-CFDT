# coding=utf-8
import numpy as np
import sys
sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402
import plott

graph = plott.Plott("../wrt/NeuNetNorm/fitness_timeline.html",
                    a_title="fitness timeline",
                    a_x_label="fitness",
                    a_y_label="time")
lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]

NeuNet = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                       layer, neurons_per_layer, 2)
NeuNet.read()

y = NeuNet.fitnessHistory

graph.add_graph(y,
                alpha=1,
                mode="line",
                legend="fitness",
                colour="blue",
                width=3)
graph.save_plott(mode="show")
