import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utilis, agent, graph


class game:
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


def stackPlot(data, r, Actions, Iterations, titleComment=""):
    A = len(Actions)
    x = range(Iterations)
    y = np.zeros((Iterations, A))
    for i in range(Iterations):
        y[i] = data[i][r]
    y = np.vstack(y.T)

    fig, ax = plt.subplots()
    # grays = np.arange(0, 1, (max(Actions) - min(Actions))/A)
    ax.stackplot(x, y, labels=Actions, colors=[str(0.9 - 0.9 * x) for x in Actions])
    ax.legend(loc='best')
    plt.ylabel('Number of Actions')
    plt.xlabel('Time(iterations)')

    title = 'Average Number of Actions in Round ' + str(r + 1)
    if titleComment:
        title += "\n" + titleComment

    plt.title(title)

    plt.savefig(titleComment + " in round " + str(r+1) + ".png")

    plt.show()


# def rep(repeat, N=100, R=1, K=99, P=0, Actions=[0, 0.2, 0.4, 0.6, 0.8], I=1000, RF=0, alpha=1, epsilon=0.1, multiArm='greedy'):

def rep(repeat=30, R=1, Actions=[0, 0.2, 0.4, 0.6, 0.8], I=1000, **kwargs):
    data = np.zeros((I, R, len(Actions)))
    Actions.sort()
    for re in range(repeat):
        print("REP", re)
        g = game(R=R, Actions=Actions, I=I, **kwargs)
        result = g.play()
        data += result
    data /= repeat
    return data


def averageOfLast(data, Actions, N=100, r=0, lastIterations=100):
    sum = 0
    action_counter = {action: 0 for action in Actions}

    for i in range(-1, -lastIterations - 1, -1):
        sum += np.sum(data[i, r] * Actions)
        for a in range(len(Actions)):
            action_counter[Actions[a]] += data[i, r, a] / lastIterations
    return (sum / (lastIterations * N), action_counter)


def graph_kp3d(Actions, Klist=[2, 4, 8, 10], Plist=[0, 0.3, 0.6, 0.9], repeat=30, N=100):
    K = Klist
    P = Plist

    meanA = np.zeros((len(K), len(P)))

    for k in range(len(K)):
        for p in range(len(P)):
            data = rep(repeat, K=K[k], P=P[p], Actions=Actions)  # Specify other params by adding here
            meanA[k][p] = averageOfLast(data, Actions, lastIterations=100, N=N)[0]  # Doing the first round only -- for now
            print("k, p, mean", k, p, meanA[k][p])

    P, K = np.meshgrid(P, K)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(P, K, meanA, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def graph3d_alpha_threshold(Actions, repeat=30, AlphaList=np.arange(0, 1.01, 0.05), ThreshList=np.arange(0.1, 1.1, 0.1), N=100, **kwargs):

    mean = np.zeros((len(ThreshList), len(AlphaList)))
    ratio_by_threshold = np.zeros((len(ThreshList), len(AlphaList)))

    for t in range(len(ThreshList)):
        for a in range(len(AlphaList)):
            print("Calculating... t, alpha = ", t, a)
            data = rep(repeat=repeat, Actions=Actions, alpha=AlphaList[a], threshold=ThreshList[t], **kwargs)
            mean[t][a] = averageOfLast(data, Actions, lastIterations=100, N=N)[0]
        ratio_by_threshold[t] = mean[t] / ThreshList[t]


    A, T = np.meshgrid(AlphaList, ThreshList)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(A, T, mean, cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Alpha')
    # ax.invert_xaxis()
    ax.set_ylabel('Threshold')
    ax.set_zlabel('Average contribution')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()


    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(A, T, ratio_by_threshold, cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    ax2.set_xlabel('Alpha')
    # ax.invert_xaxis()
    ax2.set_ylabel('Threshold')
    # ax.invert_yaxis()
    ax2.set_zlabel('Average contribution by threshold')
    fig2.colorbar(surf2, shrink=0.5, aspect=5)
    plt.show()


# def hist2d_alpha_threshold(Actions, repeat=30, AlphaList=np.arange(0, 1.1, 0.1), ThreshList=np.arange(0, 1.1, 0.2), **kwargs):




def stackBar(r, Actions, repeat=30, multiArm='greedy', **kwargs):  # Plotting the data for round r

    # if len(kwargs) != 1:
    #     print("ERROR, Stack Bar Graph Expects 1 List, gets:", len(kwargs))
    # key, alist = list(kwargs.items())[0]

    key = -1
    alist = []
    for k, v in kwargs.items():
        if isinstance(v, list):
            if key == -1:
                key = k
                alist = v
            else:
                print("ERROR, Stack Bar Graph Expects Only 1 List to Compare")
                exit(4)
    del kwargs[key]

    print("Comparing:", key)
    print("On:", alist)

    A = len(Actions)
    p = []
    count = np.zeros((A, len(alist)))  # of each action in each iter
    ind = np.arange(len(alist))
    width = 0.3

    for al in range(len(alist)):
        newKwargs = {**{key: alist[al]}, **kwargs}
        if key == 'N':
            newKwargs['K'] = alist[al] - 1
        elif 'N' not in newKwargs.keys():
            newKwargs['N'] = 100  # default value
        data = rep(repeat, Actions=Actions, multiArm=multiArm, **newKwargs) / newKwargs['N'] * 100
        action_counter = averageOfLast(data, Actions, r, 100)[1]
        for a in range(A):
            count[a, al] = action_counter[Actions[a]]
    base = 0

    for a in range(A):
        p.append(plt.bar(ind, count[a], width, bottom=base, color=str(0.9 - 0.9 * Actions[a])))
        base += count[a]

    plt.ylabel('Percentage of Actions')
    if key == 'epsilon':
        plt.xlabel(key + ' (' + multiArm + ')')
    else:
        plt.xlabel(key)
    plt.title('Average Number of Actions in Round ' + str(r + 1))
    plt.xticks(ind, alist)
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(tuple([p[x][0] for x in range(A)][::-1]), tuple(Actions[::-1]), loc='best')
    plt.show()


def main():
    # Read-in or Define Parameters

    N = 100
    R = 2
    K = 99
    P = 0
    I = 1000
    RF = 0
    alpha = 1
    Actions = [0, 0.2, 0.4, 0.6, 0.8]

    """
    Graph1: Number of Actions of Round r (start by 0) by Iteration
    """

    # RepeatTimes = 30
    # for N in [5, 10, 20, 50, 100]:
    #     K = N - 1
    #     for R in [1, 2, 4]:
    #         for alpha in [0.2, 0.4, 0.6, 0.8, 1]:
    #             data = rep(RepeatTimes, R, Actions, I, N=N, K=K, alpha=alpha)
    #             for r in range(R):
    #                 stackPlot(data, r, Actions, I, titleComment="N="+ str(N) + ", R=" + str(R) + ", alpha=" +str(alpha) + ", Well-Mixed graph")


    # for k in [2, 4, 10, 40, 90, 99]:
    #     data = rep(repeat=30, N=100, K=k, Actions=Actions, R=1, I=I, P=P)
    #     stackPlot(data, r=0, Iterations=I, Actions=Actions, titleComment=("K=" + str(k) + ", P=" + str(P)))


    # data = rep(repeat=30, Actions=Actions, R=1, I=I, RF=2, threshold=0.3)
    # stackPlot(data, r=0, Iterations=I, Actions=Actions, titleComment="threshold = 0.3")


    """
    Graph2: Average contribution by K, P
    """

    # graph_kp3d(Actions)

    """
    Graph3: Comparing a parameter (put in a list)
    """
    # stackBar(0, Actions, repeat=1, alpha=[0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # stackBar(0, Actions, repeat=30, N=[5, 10, 20, 50, 100], threshold=0.6, RF=2)
    # stackBar(0, Actions, repeat=1, RF=2, threshold=[0.2, 0.4, 0.6, 0.8, 1])

    """
    Graph4: Actions by different epsilon method + value
    """
    # stackBar(0, Actions, repeat=1, multiArm='greedy', epsilon=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # stackBar(0, Actions, repeat=30, multiArm='decrease', epsilon=[0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999])

    """
    Graph4: Average contribution by Alpha and Threshold
    """

    graph3d_alpha_threshold(Actions, repeat=30, RF=2)

if __name__ == '__main__':
    main()