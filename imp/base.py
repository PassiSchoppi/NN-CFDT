import random
from random import randint
import time
import csv
import math


def separate_data(table, timing=0, length=10):
    data = [0]*length
    counter = 0
    for p in range(timing-length, timing):
        push = table[p]
        data[counter] = push
        counter += 1
    return data
