"""
This file contains graph methods and t-test implementations. The main
function should produce all Figures and t-test results in the thesis.

Author: Liyao Zhu  liyao@student.unimelb.edu.au
Date:   Apr. 2019
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import game


def stackPlot(data, r, Actions, Iterations, legendLoc='best', titleComment=""):
    """
    Draw a stack plot from averaged data of round r.
    """
    A = len(Actions)
    x = range(Iterations)
    y = np.zeros((Iterations, A))
    for i in range(Iterations):
        y[i] = data[i][r]
    y = np.vstack(y.T)

    fig, ax = plt.subplots()
    ax.stackplot(x, y, labels=Actions, colors=[str(0.9 - 0.9 * x) for x in
                                               Actions])
    ax.legend(loc=legendLoc)
    plt.ylabel('Percentage of each action')
    plt.xlabel('Time(iterations)')

    title = 'Average Composition of Actions in Round ' + str(r + 1)
    if titleComment:
        title += "\n" + titleComment

    plt.title(title)

    # plt.savefig(titleComment + " in round " + str(r+1) + ".png")

    plt.show()


def rep(repeat=30, R=1, Actions=[0, 0.2, 0.4, 0.6, 0.8], I=1000, **kwargs):
    """
    Repeat the game over (30) trials and retrieve the average data of
    game.play()
    :return: Averaged game results, same shape as the return of game.play()
    """
    data = np.zeros((I, R, len(Actions)))
    Actions.sort()
    for re in range(repeat):
        print("REP", re)
        g = game.Game(R=R, Actions=Actions, I=I, **kwargs)
        result = g.play()
        data += result
    data /= repeat
    return data


def averageOfLast(data, Actions, N=100, r=0, lastIterations=100):
    """
    Averaged contribution and action counter of last (100) iterations from the
    data produced by rep()
    :return: a tuple: (average contribution, a dictionary as action counter)
    """
    sum = 0
    action_counter = {action: 0 for action in Actions}

    for i in range(-1, -lastIterations - 1, -1):
        sum += np.sum(data[i, r] * Actions)
        for a in range(len(Actions)):
            action_counter[Actions[a]] += data[i, r, a] / lastIterations
    return (sum / (lastIterations * N), action_counter)


def graph_kp3d(Actions, Klist=[2, 4, 8, 10], Plist=[0, 0.3, 0.6, 0.9],
               repeat=30, N=100, **kwargs):
    """
    Draw a 3D graph for graph-based model, showing the effect of K and P on
    average contributions. (No effect observed)
    """

    K = Klist
    P = Plist
    meanA = np.zeros((len(K), len(P)))

    for k in range(len(K)):
        for p in range(len(P)):
            data = rep(repeat, K=K[k], P=P[p], Actions=Actions, **kwargs)
            meanA[k][p] = averageOfLast(data, Actions, lastIterations=100,
                                        N=N)[0]
            print("k, p, mean", k, p, meanA[k][p])

    P, K = np.meshgrid(P, K)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(P, K, meanA, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def graph3d_alpha_threshold(Actions, repeat=30,
                            AlphaList=np.arange(0, 1.01, 0.05),
                            ThreshList=np.arange(0.1, 1.05, 0.05),
                            N=100, **kwargs):
    """
    Draw two 3D graphs showing the average contribution and the average
    contribution divided by threshold on two parameters: alpha and threshold
    """
    mean = np.zeros((len(ThreshList), len(AlphaList)))
    ratio_by_threshold = np.zeros((len(ThreshList), len(AlphaList)))

    for t in range(len(ThreshList)):
        for a in range(len(AlphaList)):
            print("Calculating... t, alpha = ", t, a)
            data = rep(repeat=repeat, Actions=Actions, alpha=AlphaList[a],
                       threshold=ThreshList[t], **kwargs)
            mean[t][a] = averageOfLast(data, Actions, lastIterations=100,
                                       N=N)[0]
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


def stackBar(r, Actions, repeat=30, multiArm='greedy', legendLoc='best',
             **kwargs):
    """
    Draw a stack bar graph, to compare the composition of actions on one
    parameter, specified as a list in **kwargs
    """

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
        data = rep(repeat, Actions=Actions, multiArm=multiArm, **newKwargs) /\
               newKwargs['N'] * 100
        action_counter = averageOfLast(data, Actions, r=r,
                                       lastIterations=100)[1]
        for a in range(A):
            count[a, al] = action_counter[Actions[a]]
    base = 0

    for a in range(A):
        p.append(plt.bar(ind, count[a], width, bottom=base,
                         color=str(0.9 - 0.9 * Actions[a])))
        base += count[a]

    plt.ylabel('Percentage of Actions')
    if key == 'epsilon':
        plt.xlabel(key + ' (' + multiArm + ')')
    else:
        plt.xlabel(key)
    plt.title('Average Composition of Actions in Round ' + str(r + 1))
    plt.xticks(ind, alist)
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(tuple([p[x][0] for x in range(A)][::-1]), tuple(Actions[::-1]),
               loc=legendLoc)
    plt.show()


def t_test(repeat, Actions, r=0, R=1, I=1000, lastIterations=100, N=100,
           byThreshold=False, **kwargs):
    """
    Compute the p-value of average contributions on two values of one
    parameter, specified as a tuple in **kwargs
    """
    key = -1
    atuple = ()
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            if key == -1:
                key = k
                atuple = v
            else:
                print("ERROR, T-Test Expects Only 1 Tuple to Compare")
                exit(5)
    del kwargs[key]

    samples = np.zeros((2, repeat))

    for s in (0, 1):
        newArgs = {**{key: atuple[s]}, **kwargs}
        samples[s] = repHist(repeat, Actions, R, r, I, lastIterations, N,
                             **newArgs)
        if byThreshold:
            samples[s] /= newArgs["threshold"]
        print("Sample", s, samples[s])

    print(stats.ttest_ind(samples[0], samples[1]))


def repHist(repeat, Actions, R=1, r=0, I=1000, lastIterations=100, N=100,
            **kwargs):
    """
    :return: A list of average contributions of all repetitions
    """
    hist = np.zeros(repeat)
    for re in range(repeat):
        # print("HistREP", re)
        g = game.Game(R=R, Actions=Actions, I=I, N=N, **kwargs)
        result = g.play()
        hist[re] = averageOfLast(result, Actions, N, r, lastIterations)[0]
    return hist


def main():
    # Default values

    N = 100
    R = 1
    K = 99
    P = 0
    I = 1000
    RF = 0
    alpha = 1
    Actions = [0, 0.2, 0.4, 0.6, 0.8]
    repeat = 1


    """Fig. 2"""
    data = rep(repeat=repeat, N=100, alpha=0.8, R=8)
    stackPlot(data, r=0, Iterations=I, Actions=Actions,
              legendLoc='lower right')


    """Fig. 3"""
    # stackBar(0, Actions, repeat=repeat,
    #          alpha=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #          legendLoc='lower left')

    """Fig. 4"""
    # data = rep(repeat, R=8)
    # for r in [0, 1, 3]:
    #     stackPlot(data, r, Actions, I, legendLoc='lower right')

    """Fig. 5"""
    # stackBar(0, Actions, repeat=repeat,
    #          threshold=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #          legendLoc='upper left', RF=2)

    """Fig. 6"""
    # data = rep(repeat=repeat, Actions=Actions, R=1, I=I, RF=2, threshold=0.2)
    # stackPlot(data, r=0, Iterations=I, Actions=Actions,
    #           titleComment="threshold = 0.2")

    """Fig. 7 & 8 - 3D graph comparing alpha and threshold"""
    # graph3d_alpha_threshold(Actions, repeat=repeat, RF=2)

    """Fig. 9 - simple line graph comparing alpha for threshold=0.2"""
    # alphaList = np.arange(0, 1.01, 0.02)
    # mean = np.zeros(len(alphaList))
    # for i in range(len(alphaList)):
    #     data = rep(repeat, alpha=alphaList[i], threshold=0.2, RF=2)
    #     mean[i] = averageOfLast(data, Actions)[0]
    #
    # plt.plot([0.2, 0.2], '--', color='0.5')
    # plt.plot(alphaList, mean, color='black')
    #
    # plt.ylabel('Average Contributions')
    # plt.xlabel('Alpha, the loss fraction')
    # plt.show()

    """Fig. 10 - Comparing composition on epsilon (greedy)"""
    # stackBar(0, Actions, repeat=repeat, multiArm='greedy',
    #          legendLoc='lower right',
    #          epsilon=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9])

    """Fig. 11 - Extending 0.2-greedy to 5000 iterations"""
    # data = rep(repeat=repeat, Actions=Actions, multiArm='greedy',
    #            epsilon=0.2, I=5000)
    # stackPlot(data, r=0, Iterations=5000, Actions=Actions,
    #           titleComment="0.2 - greedy", legendLoc='lower left')

    """Fig. 12 - Comparing composition on epsilon (decrease)"""
    # stackBar(0, Actions, repeat=repeat, multiArm='decrease',
    #          legendLoc='lower left',
    #          epsilon=[0.1, 0.4, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999])

    """Fig. 13 - Extending 0.999-decrease to 5000 iterations"""
    # data = rep(repeat=repeat, Actions=Actions, multiArm='decrease',
    #            epsilon=0.999, I=5000)
    # stackPlot(data, r=0, Iterations=5000, Actions=Actions,
    #           titleComment="0.999 - decrease", legendLoc='lower left')


    """T-tests 1. Average contribution of different T"""
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.9, 1.0))    #p=0.2708
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.8, 1.0))    #p=0.1096
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.7, 1.0))    #p=0.1633
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.6, 1.0))    #p=0.2208

    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.5, 1.0))    #p=2.2067e-08

    """T-test 2. Average contribution of different alpha when T=0.2"""

    # base = repHist(30, Actions, alpha=1, RF=2, threshold=0.2)
    # for alpha in np.arange(0.8, 1, 0.01):
    #     compare = repHist(30, Actions, alpha=alpha, RF=2, threshold=0.2)
    #     print("Alpha=", alpha, stats.ttest_ind(base, compare))



    """T-test 3. Avg contribution of Epsilon-decrease 0.99 with 0.1 and 0.999"""

    # base = repHist(30, Actions, multiArm='decrease', epsilon=0.99)
    # for epsilon in (0.1, 0.999):
    #     compare = repHist(30, Actions, multiArm='decrease', epsilon=epsilon)
    #     print("Epsilon=", epsilon, stats.ttest_ind(base, compare))

    """T-test 4. Avg contribution of 0.999-decrease 5000 iterations with 0.9"""
    # base = repHist(30, Actions, multiArm='decrease', epsilon=0.9)
    # compare = repHist(30, Actions, multiArm='decrease', epsilon=0.999, I=5000)
    # print(stats.ttest_ind(base, compare))


if __name__ == '__main__':
    main()