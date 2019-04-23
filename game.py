import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import utilis, agent, graph


N = 100
# K = 2
# P = 0.8
I = 1000
R = 1

Actions = [0, 0.2, 0.4, 0.6, 0.8]    # sort in ascending order


class game:
    def __init__(self, K = 2, P = 0.8 ):
        # datamap = utilis.read()
        # self.N = datamap['N']  # N-Player Game
        # self.M = datamap['M']  # Randomly choose M players to play the game (normally 2)
        # self.RF = datamap['RF']  # Parsed number of risk function chosen for the game
        # self.alpha = datamap['alpha']  # Loss fraction
        # self.R = datamap['R']   # Rounds of a game


        self.N = N
        self.M = 2
        self.RF = 0
        self.alpha = 1
        self.R = R
        self.threshold = 0.5      # Threshold
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
        IW = 100     # Initial Wealth - can be input, can be variable to distinguish population
        self.totalWealth = self.M * IW    # subject to change
        for i in range(self.N):
            self.players.append(agent.Agent(self.R,IW,self.actions))


        "Check if N is divisible by M"
        if self.N % self.M != 0:
            print("ERROR, N is not divisible by M, abort")
            exit(1)


    def lossfrac(self):
        """
            the percentage of wealth that players are going to lose if collective-risk happens
        """

        return self.alpha


    def riskfunc(self,RF,contribution,totalwealth):
        """
            the probability of collective-risk happening, given contribution
        """

        proportion = contribution/totalwealth

        if RF == 0:
            # probably parse more parameters here
            return 1 - proportion


        elif RF == 1:
            if proportion >= self.threshold:
                return 0
            else:
                return 1


        elif RF == 2:
            if proportion < self.threshold:
                return 1 - proportion / self.threshold
            else:
                return 0


        return "error"



    def computeRisk(self, contrib_sum, haveRisk = True):  ############

        if haveRisk:
            return self.riskfunc(self.RF, contrib_sum, self.totalWealth)
        else:
            return 0




    def selectPlayers(self):
        """
            Randomly select M players from population of size N for each game.
            :return: An array of permutation of players as arrays of M players
        """
        return np.random.permutation(self.N).reshape((self.N//self.M, self.M))  # A 2-dimensional array, stating index of agents


    def play2(self):

        # lastStrategyTable = np.zeros((self.N, self.R))
        # sameStrategyRounds = 0

        results = np.zeros((self.iterations,self.R, len(self.actions)))
        """ITERATION"""
        for iter in range(self.iterations):

            actionTable = np.zeros((self.N, self.R))
            strategyTable = np.zeros((self.R, self.N))        # DIFFERENT AXIS R-N
            lossTable = np.zeros((self.N, self.R))

            for playerIndex in range(self.N):                 # For each player
                player = self.players[playerIndex]
                player.resetWealth()                          # reset initial wealth
                for r in range(self.R):                              # For each round
                    action = player.chooseAction(r)
                    actionTable[playerIndex][r] = action
                    strategyTable[r][playerIndex] = player.getStrategy()[r]

            playersNo = self.graph.select()
            for r in range(self.R):
                for [i, j] in playersNo:
                    pool = 0
                    pool += self.players[i].getWealth() * actionTable[i][r] +\
                            self.players[j].getWealth() * actionTable[j][r]
                    risk = self.computeRisk(pool, self.totalWealth)

                    for p in [i, j]:
                        if np.random.uniform(0, 1) < risk:
                            lossTable[p, r] += self.lossfrac()/self.graph.getNodesNumber()[p]
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




    def playM(self):
        """
            Play an iteration of N/M games between M players, of R rounds
        """
        iteration = 1
        results = []
        last_counter = []
        same_results = 0

        # if self.graph_based:
        #     playersNo = self.graphSelect()


        while (iteration <= self.iterations) & (same_results < 50 or True):   # iteration starts

            """ITERATION"""

            iteration_counter = []

            for player in self.players:     # reset initial wealth
                player.resetWealth()


            playersNo = self.selectPlayers()
            print(playersNo)



            """GAME"""

            for m_players in playersNo:    # for each set of m players  -- a game

                game_counter = []  # STATS: list of the round counters

                print("A new game starts, among players:", m_players, "\nContribution initialised")
                contributions = {}   # accumulated contributions of each round

                """ROUND"""


                for r in range(self.R):    # for each round

                    round_counter = {}  # STATS: counting the number of each actions
                    for action in self.actions:
                        round_counter[action] = 0

                    ratio = {}
                    print("RRRRRRRRRRRRRRRRound", r)

                    """PLAYER'S TURN"""
                    for m in m_players:    # for each player
                        print("Player", m, "is playing:")
                        ratio[m] = self.players[m].chooseAction(r)         # Choose Action (a ratio)
                        round_counter[ratio[m]] += 1
                        currentWealth = self.players[m].getWealth()
                        if m not in contributions:
                            contributions[m] = 0
                        print("Ratio:", ratio[m], "current wealth before:", currentWealth)
                        contributions[m] += ratio[m] * currentWealth
                        print("Contribute: ", ratio[m] * currentWealth)
                    """PLAYER'S TURN END"""

                    print("All players contributed, sum:", sum(contributions.values()), "total wealth:", self.totalWealth)
                    risk = self.computeRisk(sum(contributions.values()), self.totalWealth)
                    print("risk:", risk)
                    for m in m_players:
                        if np.random.uniform(0,1) < risk:   # "<" since np.random.uniform is [0, 1)
                            print("XXXXXXXX Tragedy happened to Player ", m, " losing " ,self.lossfrac(), "of wealth")
                            loss = self.lossfrac()
                        else:
                            print("NOTHING HAPPEND TO PLAYER",m)
                            loss = 0
                        self.players[m].updateReward(r, ratio[m], loss)

                    print("R----------Round finished, round counter: ", round_counter)
                    game_counter.append(round_counter)
                """ROUND END"""

                print("G======Game finished. game counter:", game_counter)

                if not iteration_counter:
                    iteration_counter = copy.deepcopy(game_counter)
                else:
                    for r in range(self.R):
                        iteration_counter[r] = utilis.combine_dict(iteration_counter[r], game_counter[r])

            """GAME END"""

            print("I~~~~~Iteration ", iteration, " finished. Iteration Counter:")
            print(iteration_counter)
            results.append(iteration_counter)
            iteration += 1

            if last_counter:
                if last_counter == iteration_counter:
                    same_results += 1
                else:
                    same_results = 0

            last_counter = copy.deepcopy(iteration_counter)


        """ITERATION END"""

        print("GAME FINISHED. RESULTS:")
        for i in range(len(results)):
            print("iteration", i+1, results[i])


def stackBar(data, r):    # Plotting the data for round r

    A = len(Actions)
    p = []
    mean = np.zeros((A, I))     # of each action in each iter
    ind = np.arange(I)
    width = 0.3
    for iter in range(I):
        for a in range(A):
            mean[a, iter] = data[iter, r, a]
    base = 0
    for a in range(A):
        p.append(plt.bar(ind, mean[a], width, bottom=base))
        base += mean[a]

    plt.ylabel('Number of Actions')
    plt.xlabel('Time(iterations)')
    plt.title('Average Number of Actions in Round ' + str(r+1))
    # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend(tuple([p[x][0] for x in range(A)][::-1]), tuple(Actions[::-1]))

    plt.show()


def stackPlot(data, r, k, p):

    A = len(Actions)
    x = range(I)
    y = np.zeros((I, A))
    for i in range(I):
        y[i] = data[i][r]
    y = np.vstack(y.T)

    fig, ax = plt.subplots()
    # grays = np.arange(0, 1, (max(Actions) - min(Actions))/A)
    ax.stackplot(x, y, labels=Actions, colors=[str(1 - x) for x in Actions])
    ax.legend(loc='lower right')
    plt.ylabel('Number of Actions')
    plt.xlabel('Time(iterations)')
    plt.title('Average Number of Actions in Round ' + str(r+1) +
              '\n(k=' + str(k) + ', p=' + str(p) + ')')
    plt.show()


def rep(rept, K, P, r=0, graph = None):
    data = np.zeros((I, R, len(Actions)))
    for re in range(rept):
        print("REP", re)
        g = game(K=K, P=P)
        result = g.play2()
        data += result
    data /= rept
    print(data)


    if graph == "stackBar":
        stackBar(data, 0)

    elif graph == "stackPlot":
        if r == -1:
            for i in range(R):
                stackPlot(data, i, K, P)
        else:
            stackPlot(data, r, K, P)


    # Taking the mean of the last 100 iterations  --- need to justify

    sum = 0
    for i in range(-1, -101, -1):
        sum += np.sum(result[i, r] * Actions)
    return sum/100

def graph_kp3d(Klist=[2, 4, 8, 10], Plist=[0.2, 0.4, 0.6, 0.8], repet=30):
    K = Klist
    P = Plist


    meanA = np.zeros((len(K), len(P)))

    for k in range(len(K)):
        for p in range(len(P)):
            meanA[k][p] = rep(repet, K[k], P[p])

    P, K = np.meshgrid(P, K)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(P, K, meanA, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def main():
    graph_kp3d()
    # rep(50, 99, 0, graph="stackPlot")




if __name__ == '__main__':
    main()