import agent
import matplotlib.pyplot as plt
import numpy as np
import math
import utilis



class game():
    def __init__(self):
        datamap = utilis.read()
        self.N = datamap['N']  # N-Player Game
        self.M = datamap['M']  # Randomly choose M players to play the game (normally 2)
        self.RF = datamap['RF']  # Parsed number of risk function chosen for the game
        self.alpha = datamap['alpha']  # Loss fraction
        self.R = datamap['R']   # Rounds of a game


        self.threshold = 0      # Threshold


    def createPalyers(self):
        players = []
        IW = 100     # Initial Wealth
#        for i in range(self.N):
#            players.append(agent.Agent(self.R, self.N, IW))
#
#(strategy, wealth, fitness)


    def updatePopulation(self):
        pass

    def lossfrac(self, alpha):

        """the percentage of wealth that players are going to lose if collective-risk happens"""

        for risk in range(0,1):
           return risk


    def riskfunc(self,RF,contribution,anything):

        """the probability of collective-risk happening, given contribution"""

        if RF == 1:
            return  # probably parse more parameters here
        elif RF == 2:
            return 1
        return 0



    def computePayoff(self):
        pass


    def selectPlayers(self):
        """
            Randomly select M players from population of size N.
        """
        return np.random.choice(self.N, self.M)


    def play(self):
        pass