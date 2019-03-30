# coding=utf-8
from random import randint
import numpy as np
import sys
import os
from bokeh.plotting import figure, output_file, show, save

sys.path.insert(0, "imp/")
import base  # noqa: E402
import Course  # noqa: E402
import NeuNet  # noqa: E402
import plott  # noqa: E402

# initialising course
course_GOOG = Course.Course("GOOG", "wrt/CourseGOOG.csv")
# course_GOOG.download_course(ask=False)
course_GOOG.read_course()

training_mode = 0
neuronsPerLayer = 0
layer = 0
training_length = 0
counter = 0

while True:
    print("#################################################################")
    # creating new folder
    new_path = 'wrt/NeuNet' + str(counter)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    field = 2
    record = 500

    # changing the key dates of the AI over time
    training_mode = counter % 2
    if training_mode == 0:
        neuronsPerLayer = neuronsPerLayer + 5
    if neuronsPerLayer % 40 == 0:
        neuronsPerLayer = 5
        layer = layer + 5
    if layer % 40 == 0:
        layer = 5
        training_length = training_length + 1000

    training_mode = 0
    neuronsPerLayer = 5
    layer = 10
    training_length = 5000

    # saving key data from neural network
    key_data = np.array(["counter: "+str(counter),
                         "neurons_per_layer: " + str(neuronsPerLayer),
                         "training_mode: "+str(training_mode),
                         "layer: " + str(layer),
                         "field: " + str(field),
                         "record: " + str(record),
                         "training_length: " + str(training_length)],
                         dtype='|S')
    print(key_data)
    np.savetxt(new_path + '/key_data.txt', key_data, fmt='%s')

    # initialising course
    course_GOOG.reformat_course(days_testing=110, record_size=record)

    # initialising neural network
    AI = NeuNet.NeuNet(new_path + "/NeuNetGOOG.csv", new_path + "/NeuNetHistory.csv", layer, neuronsPerLayer, field)
    AI.reset(ask=False)
    AI.write()
    AI.read()

    # train network
    fit_history = AI.train(course_GOOG.trainData,
                           method=training_mode,
                           return_fit=True,
                           iterations=training_length)
    AI.write()

    # output
    new_path = 'wrt/NeuNet' + str(counter)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    x = np.arange(0, len(fit_history), 1)
    plott.plott(x, fit_history, new_path+"/fitness_timeline.html")

    # bokeh output
    output_file(new_path+"/predictions_and_course.html")
    p = figure(title="predictions and course",
               x_axis_label="time",
               y_axis_label="course")
    all_pred = AI.get_fitness(course_GOOG.testData, return_all=True)
    yp = all_pred
    yc = course_GOOG.testData[AI.neurons_per_layer:len(course_GOOG.testData) - 1]
    x = np.arange(0, len(all_pred), 1)
    p.line(x, yp,
           legend="predictions",
           line_width=2,
           line_color="blue")
    p.circle(x, yp, fill_color="white", line_color="blue", size=5)
    p.line(x, yc,
           legend="course",
           line_width=2,
           line_color="red")
    p.circle(x, yc, fill_color="white", line_color="red", size=5)
    p.line(x, 0, line_width=1, line_color="black")
    save(p)

    # final output
    fitness = AI.get_fitness(course_GOOG.testData)
    print("\n" + "fitness: " + str(fitness))
    new_path = 'wrt/NeuNet' + str(counter)
    with open(new_path+"/key_data.txt", "a") as file:
        file.write("b'fitness: "+str(fitness)+"'")
    data = base.separate_data(course_GOOG.allData, 0, length=AI.neurons_per_layer)
    prediction = AI.predict(data)
    print("prediction: " + str(prediction) + "\n")

    counter = counter + 1
