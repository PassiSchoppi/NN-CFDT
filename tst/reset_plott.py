# coding=utf-8
from bokeh.plotting import figure, output_file, show, save
import numpy as np
import sys
sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402
import Course  # noqa: E402
from tqdm import tqdm

lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]

# initializing bokeh, course and NeuNet
course_GOOG = Course.Course("GOOG", "../wrt/CourseGOOG.csv")
course_GOOG.read_course()
course_GOOG.reformat_course(days_testing=110, record_size=510)
AI = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                   layer, neurons_per_layer, 2)
AI.read()
output_file("../wrt/NeuNetNorm/reset.html")
p = figure(title="predictions and course",
           x_axis_label="time",
           y_axis_label="course")

y = np.array([])

for i in range(0, 500):
    AI.reset(ask=False)
    fitness = AI.get_fitness(course_GOOG.testData, mode="bin")
    y = np.append(y, fitness)
    print(fitness)
    if fitness > 20:
        AI.write()
        exit()

x = np.arange(0, len(y), 1)
p.circle(x, y, fill_color="red", line_color="red", size=15, fill_alpha=0.6)
save(p)
