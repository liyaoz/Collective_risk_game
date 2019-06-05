"""
This file contains an implementation of a 2-player R-round collective-risk game
model. Each player can choose to contribute part of their wealth to a common
pool to reduce the risk of a collective climate catastrophe. N players are
randomly paired with one another in a graph-based model in each iteration, and
play one game. Each player plays at least one game in one iteration. If N is
odd, a player could play multiple games, but the payoffs are averaged.

Author: Liyao Zhu  liyao@student.unimelb.edu.au
Date:   Apr. 2019
"""


import numpy as np
import agent, graph


class Game:
    def __init__(self, N=100, R=1, K=99, P=0, Actions=[0, 0.2, 0.4, 0.6, 0.8],
                 I=1000, RF=0, alpha=1, epsilon=0.1,multiArm='greedy',
                 threshold=0.8):
        self.N = N
        self.M = 2
        self.RF = RF
        self.alpha = alpha
        self.R = R
        self.threshold = threshold
        self.actions = Actions
        self.iterations = I

        """
        | 2-Player Game Graph Model:
        |
        | P: Probability of rewiring each original edge in the graph
        |
        | K: The number of edges(games) connected to each player. Has to be an 
        |    even number. Max k: n - 2 (for even n) | n - 1 (for odd n)
        |    k can only be odd as n - 1 (for all n). In cases k = n - 1 -> a 
        |    fully connected graph
        """
        self.rewire_p = P
        self.rewire_k = K
        # assert (self.rewire_k < self.N)
        self.graph = graph.Graph(self.N, self.rewire_k, self.rewire_p)

        "Create players"
        self.players = []
        IW = 100  # Initial Wealth
        self.totalWealth = self.M * IW
        for i in range(self.N):
            self.players.append(agent.Agent(self.R, IW, self.actions,
                                            epsilon=epsilon, multiArm=multiArm))

    def riskfunc(self, contribution, totalwealth):
        """
        Implemented different risk functions here.
        :return: the probability of disaster happening, given contribution
        and total wealth
        """

        proportion = contribution / totalwealth

        if self.RF == 0:
            # probably parse more parameters here
            return 1 - proportion


        elif self.RF == 1:
            if proportion >= self.threshold:
                return 0
            else:
                return 1


        elif self.RF == 2:

            if proportion < self.threshold:
                return 1 - proportion / self.threshold
            else:
                return 0

        return "error"

    def play(self):
        """
        Play a whole trial of I (1000) iterations, N (100) players games
        :return: a 3d numpy matrix, recording the averaged counted number of
        all actions in each round in all iterations.
        """

        results = np.zeros((self.iterations, self.R, len(self.actions)))
        """ITERATION"""
        for iter in range(self.iterations):
            # print("GAME ITERATION", iter)

            actionTable = np.zeros((self.N, self.R))
            strategyTable = np.zeros((self.R, self.N))  # DIFFERENT AXIS R-N
            lossTable = np.zeros((self.N, self.R))

            for playerIndex in range(self.N):  # For each player
                player = self.players[playerIndex]
                player.resetWealth()  # reset initial wealth
                for r in range(self.R):  # For each round
                    action = player.chooseAction(r)
                    actionTable[playerIndex][r] = action
                    strategyTable[r][playerIndex] = player.getStrategy()[r]

            playersNo = self.graph.select()
            for r in range(self.R):
                for [i, j] in playersNo:
                    pool = 0
                    pool += self.players[i].getWealth() * actionTable[i][r] + \
                            self.players[j].getWealth() * actionTable[j][r]
                    risk = self.riskfunc(pool, self.totalWealth)

                    for p in [i, j]:
                        if np.random.uniform(0, 1) < risk:
                            lossTable[p, r] += self.alpha / self.graph.getNodesNumber()[p]
                for i in range(self.N):
                    self.players[i].updateReward(r, actionTable[i][r], lossTable[i][r])

            for r in range(self.R):
                unique, count = np.unique(strategyTable[r], return_counts=True)
                round_counter = dict(zip(unique, count))
                # print("Round ", r, round_counter)

                for a in range(len(self.actions)):
                    if self.actions[a] not in round_counter:
                        pass
                    else:
                        results[iter, r, a] = round_counter[self.actions[a]]

        return results

