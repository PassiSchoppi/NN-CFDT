import base
from tqdm import tqdm
import random
from random import randint
import numpy as np


class NeuNet:
    NeuNetContent = np.array([])
    fitnessHistory = np.array([])

    def __init__(self, a_name_of_content_csv, a_name_of_history_csv, a_layer, a_neurons_per_layer, a_field):
        self.name_of_csv = str(a_name_of_content_csv)
        self.name_of_history_csv = str(a_name_of_history_csv)
        self.layer = int(a_layer)*int(a_neurons_per_layer)
        self.neurons_per_layer = int(a_neurons_per_layer)
        self.field = a_field
        self.NeuNetContent = np.full((self.neurons_per_layer, self.layer), 0)

    # sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-1.0 * x/2)) - 0.5

    @staticmethod
    def rating(real,  prediction, mode="sqr"):
        reward = real * prediction
        # linear graph
        if mode == "linear":
            return reward

        # binary graph
        if mode == "binary":
            if reward > 0:
                reward = 1
            elif reward == 0:
                reward = 0
            elif reward < 0:
                reward = -1
            return reward

        # square graph
        if mode == "sqr":
            reward = -10*(prediction - real)*(prediction - real) + abs(reward)*10
            return reward

    def read(self):
        self.NeuNetContent = np.genfromtxt(self.name_of_csv, delimiter=',')
        self.fitnessHistory = np.genfromtxt(self.name_of_history_csv)
    
    def write(self):
        # noinspection PyTypeChecker
        np.savetxt(self.name_of_csv, self.NeuNetContent, delimiter=",")
        # noinspection PyTypeChecker
        np.savetxt(self.name_of_history_csv, self.fitnessHistory)

    def reset(self, ask=True):
        tun = False
        if ask:
            if input('want to reset the neural network(y/n): ') == "y":
                tun = True
        else:
            tun = True
        if tun:
            self.fitnessHistory = np.array([])
            self.NeuNetContent = (np.random.random_sample((self.layer, self.neurons_per_layer))-0.5)*2*self.field

    def one_layer(self, layer_id):
        layer_id *= self.neurons_per_layer
        return self.NeuNetContent[layer_id:layer_id+self.neurons_per_layer]

    def predict(self, input_from_course):
        for i in range(len(input_from_course)):
            input_from_course[i] = self.sigmoid(input_from_course[i])
        synapses_of_layer = self.one_layer(0)
        first_layer = np.dot(input_from_course, synapses_of_layer)
        for i in range(1, int(self.layer/self.neurons_per_layer)):
            synapses_of_layer = self.one_layer(i)
            first_layer = np.dot(first_layer, synapses_of_layer)
            for o in range(self.neurons_per_layer):
                first_layer[o] = self.sigmoid(first_layer[o])
        return np.sum(first_layer)

    # noinspection PyUnboundLocalVariable
    def get_fitness(self, course, return_all=False, mode="pro"):
        # enable the could_be stuff for meaningful results in %
        total = 0
        could_be = 1
        if return_all:
            all_pred = np.array([])
        for time in range(self.neurons_per_layer, course.shape[0]-1):
            data = base.separate_data(course, timing=time, length=int(self.neurons_per_layer))
            real_happening = base.separate_data(course,
                                                timing=time+1,
                                                length=int(self.neurons_per_layer))[self.neurons_per_layer-1]
            prediction_from_neunet = self.predict(data)
            if return_all:
                all_pred = np.append(all_pred, prediction_from_neunet)
            if mode == "bin":
                total += self.rating(real_happening, prediction_from_neunet, mode="binary")
                could_be += self.rating(real_happening, real_happening, mode="binary")
            else:
                total += self.rating(real_happening, prediction_from_neunet)
                could_be += self.rating(real_happening, real_happening)
        if return_all:
            return all_pred
        if mode == "pro":
            total = (total/could_be)*100
        return total

    def train(self, course, method=None, return_fit=True, iterations=100):
        if (method == "synapse") or (method == 0):
            start_fitness = self.get_fitness(course)
            for _ in tqdm(range(0, iterations)):
                # mutate
                x = randint(0, self.layer-1)
                y = randint(0, self.neurons_per_layer-1)
                save = self.NeuNetContent[x][y]
                rand = random.uniform(-self.field, self.field)
                self.NeuNetContent[x][y] = self.NeuNetContent[x][y] + rand
                # check fitness
                new_fitness = self.get_fitness(course)
                if new_fitness <= start_fitness:
                    self.NeuNetContent[x][y] = save
                else:
                    start_fitness = new_fitness
                if return_fit:
                    self.fitnessHistory = np.append(self.fitnessHistory, start_fitness)

        if (method == "neuron") or (method == 1):
            start_fitness = self.get_fitness(course, mode="bin")
            for _ in tqdm(range(0, iterations)):
                # mutate
                x = randint(0, self.layer-1)
                save = self.NeuNetContent[x].copy()
                rand = random.uniform(-self.field, self.field)
                self.NeuNetContent[x] = self.NeuNetContent[x] + rand
                # check fitness
                new_fitness = self.get_fitness(course, mode="bin")
                if new_fitness <= start_fitness:
                    self.NeuNetContent[x] = save
                else:
                    start_fitness = new_fitness
                if return_fit:
                    self.fitnessHistory = np.append(self.fitnessHistory, start_fitness)

        if (method == "day") or (method == 2):
            for _ in tqdm(range(0, iterations)):
                # get dataset
                time = randint(10, len(course)-2)
                input_data = base.separate_data(course, timing=time, length=self.neurons_per_layer)
                to_predict = course[time+1]
                # check fitness
                start_fitness = self.rating(course[time+1], self.predict(input_data))
                # mutate
                x = randint(0, self.layer - 1)
                y = randint(0, self.neurons_per_layer - 1)
                save = self.NeuNetContent[x][y]
                rand = random.uniform(-self.field, self.field)
                self.NeuNetContent[x][y] = self.NeuNetContent[x][y] + rand
                # check fitness
                new_fitness = self.rating(to_predict, self.predict(input_data))
                if new_fitness <= start_fitness:
                    self.NeuNetContent[x][y] = save
                if return_fit:
                    self.fitnessHistory = np.append(self.fitnessHistory, self.get_fitness(course))

        if method is None:
            print("You didnt choose a method!!!")
        
        if return_fit:
            return self.fitnessHistory
