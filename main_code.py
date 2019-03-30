# coding=utf-8
import numpy as np
import sys
sys.path.insert(0, "imp/")
import base  # noqa: E402
import Course  # noqa: E402
import NeuNet  # noqa: E402
import plott  # noqa: E402

# key data
neuronsPerLayer = 5
layer = 10
field = 2
record = 750
training_mode = 1

# initialising course
course_GOOG = Course.Course("GOOG", "wrt/CourseGOOG.csv")
course_GOOG.download_course(ask=True)
course_GOOG.read_course()
course_GOOG.reformat_course(days_testing=110, record_size=record)

# initialising neural network
AI = NeuNet.NeuNet("wrt/NeuNetNorm/NeuNetGOOG.csv", "wrt/NeuNetNorm/NeuNetHistory.csv", layer, neuronsPerLayer, field)
AI.read()
AI.reset(ask=True)
AI.write()

# train network
training_length = int(input("iterations training(int): "))
fit_history = AI.train(course_GOOG.trainData,
                       method=training_mode,
                       return_fit=True,
                       iterations=training_length)
AI.write()

# saving key data from neural network
key_data = np.array(["neurons_per_layer: " + str(neuronsPerLayer),
                     "training_mode: " + str(training_mode),
                     "layer: " + str(layer),
                     "field: " + str(field),
                     "record: " + str(record),
                     "training_length: " + str(len(AI.fitnessHistory))],
                     dtype='|S')
print(key_data)
np.savetxt('wrt/NeuNetNorm/key_data.txt', key_data, fmt='%s')

# output
data = base.separate_data(course_GOOG.allData, 0, length=AI.neurons_per_layer)
prediction = AI.predict(data)
print("prediction: "+str(prediction)+"\n")
