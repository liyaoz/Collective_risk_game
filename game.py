import agent
import matplotlib.pyplot as plt
import numpy as np
import math
import utilis



class game():
    def __init__(self):
        datamap = utilis.read()
        self.N = datamap['N']