# coding=utf-8
import numpy as np
import sys
sys.path.insert(0, "../imp/")
import plott

graph = plott.Plott("../wrt/NeuNetNorm/plott_test.html", a_title="testTitle", a_x_label="xTitle", a_y_label="yTitle")

graph.add_graph(np.arange(0, 10, 1))

graph.save_plott(mode="show")
