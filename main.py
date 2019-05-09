import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
from scipy import stats
import game


def stackPlot(data, r, Actions, Iterations, titleComment=""):
    A = len(Actions)
    x = range(Iterations)
    y = np.zeros((Iterations, A))
    for i in range(Iterations):
        y[i] = data[i][r]
    y = np.vstack(y.T)

    fig, ax = plt.subplots()
    ax.stackplot(x, y, labels=Actions, colors=[str(0.9 - 0.9 * x) for x in Actions])
    ax.legend(loc='best')
    plt.ylabel('Number of Actions')
    plt.xlabel('Time(iterations)')

    title = 'Average Number of Actions in Round ' + str(r + 1)
    if titleComment:
        title += "\n" + titleComment

    plt.title(title)

    # plt.savefig(titleComment + " in round " + str(r+1) + ".png")

    plt.show()


def rep(repeat=30, R=1, Actions=[0, 0.2, 0.4, 0.6, 0.8], I=1000, **kwargs):
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
        action_counter = averageOfLast(data, Actions, r=r, lastIterations=100)[1]
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


def t_test(repeat, Actions, r=0, R=1, I=1000, lastIterations=100, N=100, byThreshold=False, **kwargs):
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
        # for re in range(repeat):
        #     # print("T-Test REP", re)
        #     g = game.Game(R=R, Actions=Actions, I=I, N=N, **newArgs)
        #     result = g.play()
        #     samples[s, re] = averageOfLast(result, Actions, N, r, lastIterations)[0]
        samples[s] = repHist(repeat, Actions, R, r, I, lastIterations, N, **newArgs)
        if byThreshold:
            samples[s] /= newArgs["threshold"]
        print("Sample", s, samples[s])

    print(stats.ttest_ind(samples[0], samples[1]))


def repHist(repeat, Actions, R=1, r=0, I=1000, lastIterations=100, N=100, **kwargs):
    hist = np.zeros(repeat)
    for re in range(repeat):
        print("HistREP", re)
        g = game.Game(R=R, Actions=Actions, I=I, N=N, **kwargs)
        result = g.play()
        hist[re] = averageOfLast(result, Actions, N, r, lastIterations)[0]
    return hist




def main():
    # Read-in or Define Parameters

    N = 100
    R = 1
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


    # for k in [2, 99]:
    #     for p in [0.8]:
    #         data = rep(repeat=30, N=100, K=k, Actions=Actions, R=1, I=I, P=p)
    #         stackPlot(data, r=0, Iterations=I, Actions=Actions, titleComment=("K=" + str(k) + ", P=" + str(p)))


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
    # stackBar(0, Actions, repeat=1, N=[5, 10, 20, 50, 100], threshold=0.6, RF=2)
    # stackBar(0, Actions, repeat=1, RF=2, threshold=[0.2, 0.4, 0.6, 0.8, 1])

    """
    Graph4: Actions by different epsilon method + value
    """
    # stackBar(0, Actions, repeat=1, multiArm='greedy', epsilon=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # stackBar(0, Actions, repeat=1, multiArm='decrease', epsilon=[0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999])

    """
    Graph5: Average contribution by Alpha and Threshold
    """

    # graph3d_alpha_threshold(Actions, repeat=1, RF=2)

    """
    T-Test
    """

    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.2, 0.3), byThreshold=True)   #p=3.324e-31
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.6, 1.0))    #pvalue=0.2208
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.8, 1.0))    #pvalue=0.1096
    # t_test(30, Actions, alpha=1, RF=2, threshold=(0.5, 1.0))    #pvalue=2.2067e-08
    # t_test(30, Actions, alpha=0.85, RF=2, threshold=(0.2, 0.3), byThreshold=True)   #pvalue=0.005865

    # t_test(30, Actions, alpha=(1, 0.9), RF=2, threshold=0.2)    #pvalue=0.3748
    # t_test(30, Actions, alpha=(1, 0.85), RF=2, threshold=0.2)   #pvalue=0.001466
    # t_test(30, Actions, alpha=(1, 0.8), RF=2, threshold=0.2)    #pvalue=0.0002030
    # t_test(30, Actions, alpha=(1, 0.75), RF=2, threshold=0.2)   #pvalue=3.9617e-07
    # t_test(30, Actions, alpha=(1, 0.7), RF=2, threshold=0.2)    #pvalue=2.2428e-09
    # t_test(30, Actions, alpha=(1, 0.65), RF=2, threshold=0.2)   #pvalue=6.8966e-09
    # t_test(30, Actions, alpha=(1, 0.6), RF=2, threshold=0.2)    #pvalue=7.1621e-15
    # t_test(30, Actions, alpha=(1, 0.5), RF=2, threshold=0.2)    #pvalue=4.1760e-13
    # t_test(30, Actions, alpha=(1, 0.45), RF=2, threshold=0.2)   #pvalue=1.3749e-11
    # t_test(30, Actions, alpha=(1, 0.4), RF=2, threshold=0.2)    #pvalue=3.8352e-19

    """T-TEST GRAPH"""

    # t_test(30, Actions, K=(2, 99), P=0)    #pvalue=0.4278
    # t_test(30, Actions, K=(2, 99), P=0.9)  #pvalue=0.4541
    # t_test(100, Actions, K=(2, 99), P=0.8) #pvalue=0.01502  ***
    # t_test(30, Actions, K=(2, 99), P=0.85) #pvalue=0.1931
    # t_test(30, Actions, K=(2, 99), P=0.75) #pvalue=0.5630
    # t_test(30, Actions, K=2, P=(0, 0.9))   #pvalue=0.9806
    # t_test(30, Actions, K=2, P=(0, 0.8))   #pvalue=0.4523
    # t_test(30, Actions, K=(2, 99), P=0.9)  #pvalue=0.4541
    # t_test(30, Actions, K=(2, 99), P=0.7)  #pvalue=0.3698



if __name__ == '__main__':
    main()