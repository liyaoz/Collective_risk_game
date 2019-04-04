# NOTES:
#


# Imports ______________________________________________________

import matplotlib.pyplot as plt
import numpy as np
import math

# Constants ____________________________________________________

N = 100  # Size of the population
M = 6  # Number of individuals (from the population)
R = 10  # Total number of rounds in one game
T = M * R  # Target Sum
GEN = 10 # Number of generations
G = 1000  # Number of games per generation
PRISK = 0.9
BETA = 1  # Used in fitness (measures the intensity of selection)
MU = 0.03  # Error probability
SIGMA = 0.15  # Standard deviation for Gaussian noise on thresholds
DELTA = 0.0 # Interest

# Code _________________________________________________________

class Collective_Risk_Dilemma():
    def __init__(self, prisk=PRISK, sigma=SIGMA):
        self.prisk = prisk
        self.sigma = sigma

        self.payoffs = [[] for i in range(N)]
        self.fitness = [0] * N
        self.commonPool = 0
        self.moneyGiven = [[] for i in range(M)]


        #这个东西是用来统计 多个玩家的游戏中，一次游戏（不是一轮）结束后，系统的期待应该是
        # 每个人都能在每一轮投入1 那么应该所有人都是 cEqualsR 实际不可能， 就看看实际的情况
        # 有多少人是 free rider ，投入了但是不够， fair share, 还有无私利他的人
        self.cContributions = [0]*4 #[cEqualsZero, cEqualsR, cBiggerThanR, cSmallerThanR]
        # Set Players Strategies:
        self.contributions = np.random.choice(3, size=(N,R,2))  # 0 Defectors, 1 Fair Sharers, 2 Altruists

        # 结果是一个大tuple 里面有 N个小tuple  每个小tuple里面有R个元素  每个元素都是0-1之间
        self.threshold = np.random.random(size=(N,R))
        #for n in range(N):
         #   self.contributions[n] = np.random.choice(3, (R, 2))
         #   self.threshold[n] = np.random.random(size=R)

    def ComputePayoff(self, player, invested):
        """
            At the end of each game, the payoff must be recalculated
        """
        if self.commonPool >= T or np.random.random() < np.random.choice([True, False], 1, True, [1 - self.prisk, self.prisk]):
            return (2 * R) - invested
        else:
            return 0

    def SelectPlayers(self):
        """
            Randomly select M players from population of size N.
        """
        return np.random.choice(N, M)

    def UpdatePopulation(self):
        """
            The next generation is selected using the Wright-Fisher process
            where the individual’s fitness is used to weigh the probability of
            choosing an individual for the new population.
        """
        prob = list(map(lambda x: x / sum(self.fitness), self.fitness))
        reproduction_selection = np.random.choice(N, N, True, prob)
        temp_contributions = np.zeros((N,R,2))
        temp_threshold = np.zeros((N,R))
        for i, player in enumerate(reproduction_selection):  # Generate errors
            for r in range(R):
                if np.random.random() < MU:  # Error on contribution 1
                    temp_contributions[i,r,0] = np.random.choice(3)
                else:
                    temp_contributions[i,r,0] = np.copy(self.contributions[player,r,0])
                if np.random.random() < MU:  # Error on contribution 2
                    temp_contributions[i,r,1] = np.random.choice(3)
                else:
                    temp_contributions[i,r,1] = np.copy(self.contributions[player,r,1])
                if np.random.random() < MU:  # Error on threshold
                    temp_threshold[i][r] = np.random.normal(self.threshold[player,r], self.sigma)
                else:
                    temp_threshold[i][r] = np.copy(self.threshold[player,r])
        self.contributions = np.copy(temp_contributions)
        self.threshold = np.copy(temp_threshold)
        self.payoffs = [[] for i in range(N)]

    def Play(self):
        """
            Plays 1 Game  (R rounds)
        """
        selectedPlayers = self.SelectPlayers()  # Index of the player in the whole population (M)
        moneyPlayerOwns = [2 * R] * M  # An individual player starts each game with an initial endowment of 2R
        c = [0] * M  # Total Investment of each player
        self.moneyGiven = [[] for i in range(M)]   # 要M个 []
        self.commonPool = 0
        for r in range(R):  # 对每一轮游戏来说
            if self.commonPool < T: # If the goal was not reached yet: Put more money; Else: Do Nothing
                for i, player in enumerate(selectedPlayers):
                    if self.commonPool / T < self.threshold[player,r]:  # threshold not attained yet
                        if self.contributions[player,r,0] <= moneyPlayerOwns[i]:  # enough money to put in the commonpool
                            c[i] += self.contributions[player,r,0]
                            self.commonPool += self.contributions[player,r,0]
                            self.moneyGiven[i].append(self.contributions[player,r,0])
                            moneyPlayerOwns[i] -= self.contributions[player,r,0]
                        else:  # Else: Put what's left
                            c[i] += moneyPlayerOwns[i]
                            self.moneyGiven[i].append(moneyPlayerOwns[i])
                            self.commonPool += moneyPlayerOwns[i]
                            moneyPlayerOwns[i] = 0

                    else:  # threshold was attained
                        if self.contributions[player,r,1] <= moneyPlayerOwns[i]:  # enough money to put in the commonpool
                            c[i] += self.contributions[player,r,1]
                            self.commonPool += self.contributions[player,r,1]
                            self.moneyGiven[i].append(self.contributions[player,r,1])
                            moneyPlayerOwns[i] -= self.contributions[player,r,1]
                        else:  # Else: Put what's left
                            c[i] += moneyPlayerOwns[i]
                            self.moneyGiven[i].append(moneyPlayerOwns[i])
                            self.commonPool += moneyPlayerOwns[i]
                            moneyPlayerOwns[i] = 0
            self.commonPool += self.commonPool*DELTA

        for i, player in enumerate(selectedPlayers):
            self.payoffs[player].append(self.ComputePayoff(i, c[i]))

        # cContributions = [cEqualsZero, cEqualsR, cBiggerThanR, cSmallerThanR]
        for i in range(M):
            if c[i] == 0:
                self.cContributions[0] += 1
            elif c[i] == R:
                self.cContributions[1] += 1
            elif c[i] > R:
                self.cContributions[2] += 1
            elif c[i] < R:
                self.cContributions[3] += 1
            else: raise Exception


        if T < self.commonPool: return 1 # If target was reached
        else: return 0







    #def GetPayoffs(self):
    #    return self.payoffs


    def GetAverageFitness(self):
        return np.mean(self.fitness)


    def GetAveragePayoff(self):
        payoff_sum = 0
        for n in range(N):
            payoff_sum += np.mean(self.payoffs[n])
        return np.mean(payoff_sum)

    def GetCommonPool(self):
        return self.commonPool

    def GetcContributions(self):
        return self.cContributions



    def UpdateFitness(self):
        """
            Receives a list of all the payoffs
            of a player. Computes the average
            and returns the exp(Bpii)
        """
        for n in range(N):
            self.fitness[n] = math.exp(BETA * np.mean(self.payoffs[n]))




# Plots ________________________________________________________


def SampleTrajectories():
    # Variables for the plots:
    ratioAveragePayoff  = []
    ratioTargetReached  = []
    ratioAverageContribution   = []

    game = Collective_Risk_Dilemma()  # Initialize one simulation
    for g in range(GEN):
        print("Playing Generation", g)
        timesReachedTarget = 0
        for i in range(G):
            timesReachedTarget += game.Play()  # Play one game (10 rounds)       

        ratioAveragePayoff.append(game.GetAveragePayoff() / (2*G)) # 2*G is the max payoff (to obtain a value between 0 and 1)
        ratioTargetReached.append(timesReachedTarget / G)
        ratioAverageContribution.append(game.GetCommonPool() / (2*N))

        game.UpdateFitness()
        game.UpdatePopulation()  # Create a new population based on fitness

    plt.plot([x for x in range(len(ratioAveragePayoff))], ratioAveragePayoff, label="Payoff")
    plt.plot([x for x in range(len(ratioTargetReached))], ratioTargetReached, label="Target Reached")
    plt.plot([x for x in range(len(ratioAverageContribution))], ratioAverageContribution, label="Contribution")
    plt.xlabel('Generation')
    plt.ylabel('Proportion')
    plt.title('Sample trajectories for the evolutionary dynamics in collective-risk dilemmas')
    plt.legend()
    plt.show()








def SummaryEvolutionaryDynamics1():

    risks = [i*(1/20) for i in range(21)]
    payoffRisk = []
    contributionRisk = []
    targetRisk = []
    firstHalf = []
    secondHalf = []

    for k in risks:
        print("PRISK", k)
        ratioPayoff = []
        ratioTarget = []
        ratioContribution = []
        game = Collective_Risk_Dilemma(k) # (PRISK, SIGMA)
        for g in range(GEN):
            print("Playing Generation", g)
            timesReachedTarget = 0
            for i in range(G):
                timesReachedTarget += game.Play()  # Play one game (10 rounds) 

            ratioPayoff.append(game.GetAveragePayoff() / (2*G))
            ratioTarget.append(timesReachedTarget / G)
            ratioContribution.append(game.GetCommonPool() / (2*N))

            game.UpdateFitness()
            game.UpdatePopulation()  # Create a new population based on fitness

        payoffRisk.append(np.mean(ratioPayoff))
        contributionRisk.append(np.mean(ratioContribution))
        targetRisk.append(np.mean(ratioTarget))
        firstHalf.append(np.mean(ratioContribution[:len(ratioContribution)//2]))
        secondHalf.append(np.mean(ratioContribution[len(ratioContribution)//2:]))

    plt.plot(risks, payoffRisk, label="Payoff")
    plt.plot(risks, contributionRisk, label="Contribution")
    plt.plot(risks, targetRisk, label="Target")
    plt.plot(risks, firstHalf, label="1st Half")
    plt.plot(risks, secondHalf, label="2nd Half")
    plt.xlabel('Risk Probability')
    plt.ylabel('Proportion')
    plt.title('Summary of the evolutionary dynamics in collective-risk dilemmas')
    plt.legend()
    plt.show()






def SummaryEvolutionaryDynamics2():

    risks = [i*(1/20) for i in range(21)]
    cEqualsZero, cEqualsR, cBiggerThanR, cSmallerThanR = [],[],[],[]

    for k in risks:
        print("PRISK", k)
        game = Collective_Risk_Dilemma(k) # (PRISK, SIGMA)
        for g in range(GEN):
            print("Playing Generation", g)
            timesReachedTarget = 0
            for i in range(G):
                timesReachedTarget += game.Play()  # Play one game (10 rounds)

            game.UpdateFitness()
            game.UpdatePopulation()  # Create a new population based on fitness

        # cContributions = [cEqualsZero, cEqualsR, cBiggerThanR, cSmallerThanR]
        contribution = game.GetcContributions()
        totalContributions = 0
        for cont in contribution:
            totalContributions += cont

        cEqualsZero.append(contribution[0]/totalContributions)
        cEqualsR.append(contribution[1]/totalContributions)
        cBiggerThanR.append(contribution[2]/totalContributions)
        cSmallerThanR.append(contribution[3]/totalContributions)
        
    plt.plot(risks, cEqualsZero, label="C = 0")
    plt.plot(risks, cEqualsR, label="C = R")
    plt.plot(risks, cBiggerThanR, label="C > R")
    plt.plot(risks, cSmallerThanR, label="C < R")
    plt.xlabel('Risk Probability')
    plt.ylabel('Frequency of the Behaviour')
    plt.title('Summary of the evolutionary dynamics in collective-risk dilemmas')
    plt.legend()
    plt.show()






# ______________________________________________________________

def main():
    #SampleTrajectories()
    # SummaryEvolutionaryDynamics1()
    SummaryEvolutionaryDynamics2()



if __name__ == '__main__':
    main()


