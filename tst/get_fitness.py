# coding=utf-8
import numpy as np
import sys
sys.path.insert(0, "../imp/")
import NeuNet  # noqa: E402
import Course  # noqa: E402

lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]
neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]
records = str(lines[4])[10:len(lines[4])-1]

course_GOOG = Course.Course("GOOG", "../wrt/CourseGOOG.csv")
course_GOOG.read_course()
course_GOOG.reformat_course(days_testing=110, record_size=records)
NeuNet = NeuNet.NeuNet("../wrt/NeuNetNorm/NeuNetGOOG.csv", "../wrt/NeuNetNorm/NeuNetHistory.csv",
                       layer, neurons_per_layer, 2)
NeuNet.read()

output = NeuNet.get_fitness(course_GOOG.testData)
print("fitness(%): "+str(output))

output = NeuNet.get_fitness(course_GOOG.testData, mode="bin")
print("fitness(10): "+str(output))
