"""
This file contains an agent class, where agents are using Q-learning to evolve
strategy to play in a collective-risk game. For multi-arm bandit, epsilon-
greedy is implemented. Agents don't recognise their opponent, nor have memory
of previous rounds of a game. Their actions are based solely on their own
Q-Table, where states are consisted of round numbers and available actions.

Author: Liyao Zhu  liyao@student.unimelb.edu.au
Date:   Apr. 2019
"""


import numpy as np


class Agent:

    def __init__(self, rounds, initialWealth, availableActions, alpha=0.1,
                 gamma=0.9, epsilon=0.1):

        self.R = rounds
        self.initialWealth = initialWealth
        self.wealth = initialWealth
        self.availableActions = availableActions
        # self.iteration = 0

        """initialise Q table to small random numbers"""
        self.qTable = np.random.rand(self.R, len(self.availableActions)) * 0.01

        "Q-Learning Parameters"
        self.learnRate = alpha
        self.discount  = gamma
        self.epsilon   = epsilon

    def updateReward(self, round, action, loss):
        """
        :param round:
        :param action:
        :param loss:
        :return:
        """
        newWealth = self.wealth * (1-action) * (1-loss)
        reward = newWealth - self.wealth
        self.wealth = newWealth

        index = self.availableActions.index(action)
        if round == self.R - 1:
            """ at goal state, no future value"""
            maxNextQ = 0
        elif round < self.R - 1:
            """ not at goal state"""
            maxNextQ = max(self.qTable[round + 1])
        else:
            print("ERROR: Illegal round number")
            exit(2)
        self.qTable[round][index] += self.learnRate * (
                reward + self.discount * maxNextQ - self.qTable[round][index])
        # print("QTABLE:", self.qTable)

        # if round == self.R - 1:
        #     self.iteration += 1
            # print("Player iteration +1 =", self.iteration)


    def chooseAction(self, roundNumber):

        """Method: Q-learning"""

        """Epsilon Decrease"""

        # if np.random.uniform(0, 1) <= 1 * self.epsilon ** self.iteration:

        """EPSILON GREEDY"""

        if np.random.uniform(0, 1) <= self.epsilon:

            return np.random.choice(self.availableActions)
        else:
            index = np.argmax(self.qTable[roundNumber])
            return self.availableActions[index]

    def getStrategy(self):
        """
        Get the current strategy without randomness, for analytical use
        :return: a dictionary of actions in all rounds
        """
        strategy = {}
        for r in range(self.R):
            index = np.argmax(self.qTable[r])
            strategy[r] = self.availableActions[index]
        return strategy


    def getWealth(self):
        return self.wealth

    def resetWealth(self):
        self.wealth = self.initialWealth