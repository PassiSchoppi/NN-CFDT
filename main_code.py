# coding=utf-8
import numpy as np
import sys
sys.path.insert(0, "imp/")
import base  # noqa: E402
import Course  # noqa: E402
import NeuNet  # noqa: E402
import plott  # noqa: E402

# key data
neuronsPerLayer = 15
layer = 5
field = 2
record = 1500
training_mode = 1

# initialising course
course_GOOG = Course.Course("GOOG", "wrt/CourseGOOG.csv")
if input("want to download latest course(y/n): ") == "y":
    course_GOOG.download_course()
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
                     "training_length: " + str(training_length)],
                    dtype='|S')
print(key_data)
np.savetxt('wrt/NeuNetNorm/key_data.txt', key_data, fmt='%s')

# output
x = np.arange(0, len(fit_history), 1)
plott.plott(x, fit_history, "wrt/NeuNetNorm/fitness_timeline.html")
print("\n" + "fitness: " + str(AI.get_fitness(course_GOOG.testData, mode="bin")))
data = base.separate_data(course_GOOG.allData, 0, length=AI.neurons_per_layer)
prediction = AI.predict(data)
print("prediction: "+str(prediction)+"\n")
