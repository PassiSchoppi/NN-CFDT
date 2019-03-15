# coding=utf-8
from bokeh.plotting import figure, output_file, show, save
import numpy as np
import sys
sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402
import Course  # noqa: E402

lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]

# initializing bokeh, course and NeuNet
course_GOOG = Course.Course("GOOG", "../wrt/CourseGOOG.csv")
course_GOOG.read_course()
course_GOOG.reformat_course(days_testing=110, record_size=1500)
AI = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                   layer, neurons_per_layer, 2)
AI.read()
output_file("../wrt/NeuNetNorm/predictions_and_course.html")
p = figure(title="predictions and course",
           x_axis_label="time",
           y_axis_label="course")

all_pred = AI.get_fitness(course_GOOG.testData, return_all=True)
yp = all_pred
yc = course_GOOG.testData[AI.neurons_per_layer:len(course_GOOG.testData) - 1]
x = np.arange(0, len(all_pred), 1)
p.line(x, yc,
       legend="course",
       line_width=2,
       line_color="red")
p.circle(x, yc, fill_color="white", line_color="red", size=5)
p.line(x, yp,
       legend="predictions",
       line_width=2,
       line_color="blue")
p.circle(x, yp, fill_color="white", line_color="blue", size=5)
p.line(x, 0, line_width=1, line_color="black")
save(p)
