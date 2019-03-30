import datetime
import pandas_datareader as web
import numpy as np
from numpy import genfromtxt


class Course:
    rawData = np.array((0, 0))
    allData = np.array((0, 0))
    trainData = np.array((0, 0))
    testData = np.array((0, 0))

    def __init__(self, a_course_tag, a_table_name):
        self.courseTag = a_course_tag
        self.tableName = a_table_name

    # most likely for offline use
    def read_course(self):
        #self.rawData = np.genfromtxt(str(self.tableName), delimiter=",")
        self.rawData = genfromtxt(self.tableName, delimiter=",")

    def write_course(self):
        np.savetxt(str(self.tableName), self.rawData, delimiter=",")

    def download_course(self, ask=True):
        if ask:
            if input("want to download latest course(y/n): ") == "y":
                print("downloading...")
                self.rawData = web.get_data_yahoo(str(self.courseTag))
                self.rawData.to_csv(path_or_buf=str(self.tableName))
                self.read_course()
        else:
            print("downloading...")
            self.rawData = web.get_data_yahoo(str(self.courseTag))
            self.rawData.to_csv(path_or_buf=str(self.tableName))
            self.read_course()

    
    def reformat_course(self, days_testing=110, record_size=510, plot=False):
        self.allData = []
        for i in range(1, len(self.rawData)):
            self.allData = np.append(self.allData, (float(self.rawData[i][4]) / float(self.rawData[i][3]))-1)
        self.allData = self.allData[len(self.allData)-record_size:len(self.allData)]
        self.trainData = self.allData[0: len(self.allData)-days_testing]
        self.testData = self.allData[len(self.allData)-days_testing: len(self.allData)]
