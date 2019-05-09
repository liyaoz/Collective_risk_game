import numpy as np
import agent, graph


class Game:
    def __init__(self, N=100, R=1, K=99, P=0, Actions=[0, 0.2, 0.4, 0.6, 0.8], I=1000, RF=0, alpha=1, epsilon=0.1,
                 multiArm='greedy', threshold=0.8):
        # datamap = utilis.read()
        # self.N = datamap['N']  # N-Player Game
        # self.M = datamap['M']  # Randomly choose M players to play the game (normally 2)
        # self.RF = datamap['RF']  # Parsed number of risk function chosen for the game
        # self.alpha = datamap['alpha']  # Loss fraction
        # self.R = datamap['R']   # Rounds of a game

        self.N = N
        self.M = 2
        self.RF = RF
        self.alpha = alpha
        self.R = R
        self.threshold = threshold  # Threshold
        # self.actions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.actions = Actions
        self.iterations = I

        """
        | 2-Player Game Graph Model:
        |
        | P: Probability of rewiring each original edge in the graph
        |
        | K: The number of edges(games) connected to each player. Has to be an even number.
        |    If 1 is desired, don't use graph model. Max k: n - 2 (for even n) | n - 1 (for odd n)
        |    * k can be odd as n - 1 (for even n). In cases k = n - 1 (for all n) -> a fully connected graph
        """
        # self.graph_based = True
        self.rewire_p = P
        self.rewire_k = K
        # assert (self.rewire_k < self.N)
        self.graph = graph.Graph(self.N, self.rewire_k, self.rewire_p)

        "Create players"
        self.players = []
        IW = 100  # Initial Wealth - can be input, can be variable to distinguish population
        self.totalWealth = self.M * IW  # subject to change
        for i in range(self.N):
            self.players.append(agent.Agent(self.R, IW, self.actions, epsilon=epsilon, multiArm=multiArm))

    def riskfunc(self, contribution, totalwealth):
        """
            the probability of collective-risk happening, given contribution
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

        # lastStrategyTable = np.zeros((self.N, self.R))
        # sameStrategyRounds = 0

        results = np.zeros((self.iterations, self.R, len(self.actions)))
        """ITERATION"""
        for iter in range(self.iterations):

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

            """Strategy Stats"""
            # if np.array_equal(strategyTable, lastStrategyTable):
            #     sameStrategyRounds += 1
            # else:
            #     sameStrategyRounds = 0
            #     lastStrategyTable = strategyTable

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

