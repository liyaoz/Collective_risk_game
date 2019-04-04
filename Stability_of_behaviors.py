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
GEN = 200 # Number of generations
G = 1000  # Number of games per generation
PRISK = 0.9
BETA = 1  # Used in fitness (measures the intensity of selection)
MU = 0.03  # Error probability
SIGMA = 0.15  # Standard deviation for Gaussian noise on thresholds

# Code _________________________________________________________

class Collective_Risk_Dilemma():
    def __init__(self, robustness=False, prisk=PRISK, sigma=SIGMA):
        self.prisk = prisk
        self.sigma = sigma

        self.payoffs = [[] for i in range(N)]
        self.fitness = [0] * N
        self.commonPool = 0
        self.moneyGiven = [[] for i in range(M)]
        self.cContributions = [0]*4 #[cEqualsZero, cEqualsR, cBiggerThanR, cSmallerThanR]
        # Set Players Strategies:

        self.contributions = []
        for i in range(N):
            self.contributions.append([])
            choice = np.random.choice(3)
            for k in range(R//2):
                self.contributions[i].append(choice)
            choice = np.random.choice(3)
            for l in range(R-R//2):
                self.contributions[i].append(choice)


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
        temp_contributions = np.zeros((N,R,1))
        temp_threshold = np.zeros((N,R))
        for i, player in enumerate(reproduction_selection):  # Generate errors
            for r in range(R):
                if np.random.random() < MU:  # Error on contribution 1
                    temp_contributions[i][r] = np.random.choice(3)
                else:
                    temp_contributions[i][r] = np.copy(self.contributions[player][r])
        self.contributions = np.copy(temp_contributions)
        self.payoffs = [[] for i in range(N)]

    def Play(self):
        """
            Plays 1 Game  (R rounds)
        """
        selectedPlayers = self.SelectPlayers()  # Index of the player in the whole population (M)
        moneyPlayerOwns = [2 * R] * M  # An individual player starts each game with an initial endowment of 2R
        c = [0] * M  # Total Investment of each player
        self.moneyGiven = [[] for i in range(M)]
        self.commonPool = 0
        for r in range(R):
            if self.commonPool < T: # If the goal was not reached yet: Put more money; Else: Do Nothing
                for i, player in enumerate(selectedPlayers):
                    if self.contributions[player][r] <= moneyPlayerOwns[i]:  # enough money to put in the commonpool
                        c[i] += self.contributions[player][r]
                        self.commonPool += self.contributions[player][r]
                        self.moneyGiven[i].append(self.contributions[player][r])
                        moneyPlayerOwns[i] -= self.contributions[player][r]
                    else:  # Else: Put what's left
                        c[i] += moneyPlayerOwns[i]
                        self.moneyGiven[i].append(moneyPlayerOwns[i])
                        self.commonPool += moneyPlayerOwns[i]
                        moneyPlayerOwns[i] = 0


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


    def UpdateFitness(self):
        """
            Receives a list of all the payoffs
            of a player. Computes the average
            and returns the exp(Bpii)
        """
        for n in range(N):
            self.fitness[n] = math.exp(BETA * np.mean(self.payoffs[n]))


    def StillStrategy00(self):
        return [0]*R in self.contributions

    def StillStrategy11(self):
        return [1]*R in self.contributions

    def StillStrategy22(self):
        return [2]*R in self.contributions

    def StillStrategy02(self):
        return [0]*(R//2)+[2]*(R//2) in self.contributions

    def StillStrategy20(self):
        return [2]*(R//2)+[0]*(R//2) in self.contributions

    
            



# Plots ________________________________________________________


def plot():
    risks = [i*(1/20) for i in range(21)]

    cntl00, cntl11, cntl22, cntl02, cntl20 = [],[],[],[],[]


    for k in risks:
        flag00, flag11, flag22, flag02, flag20 = True, True, True, True, True
        cnt00, cnt11, cnt22, cnt02, cnt20 = 0, 0, 0, 0, 0
        print("PRISK", k)
        game = Collective_Risk_Dilemma(k) # (PRISK, SIGMA)
        for g in range(GEN):
            print("Playing Generation", g)
            timesReachedTarget = 0
            for i in range(G):
                timesReachedTarget += game.Play()  # Play one game (10 rounds)

            if flag00 and game.StillStrategy00(): cnt00 += 1
            else: flag00 = False

            if flag11 and game.StillStrategy11(): cnt11 += 1
            else: flag11 = False

            if flag22 and game.StillStrategy22(): cnt22 += 1
            else: flag22 = False

            if flag02 and game.StillStrategy02(): cnt02 += 1
            else: flag02 = False

            if flag20 and game.StillStrategy20(): cnt20 += 1
            else: flag20 = False

            game.UpdateFitness()
            game.UpdatePopulation()  # Create a new population based on fitness

        cntl00.append(cnt00)
        cntl11.append(cnt11)
        cntl22.append(cnt22)
        cntl02.append(cnt02)
        cntl20.append(cnt20)


    plt.plot(risks, cntl00, label="C = 0000000000")
    plt.plot(risks, cntl11, label="C = 1111111111")
    plt.plot(risks, cntl22, label="C = 2222222222")
    plt.plot(risks, cntl02, label="C = 0000022222")
    plt.plot(risks, cntl20, label="C = 2222200000")


    plt.xlabel('Risk Probability')
    plt.ylabel('Generations')
    plt.title('Stability of behaviors in a collective-risk dilemma')
    plt.legend()
    plt.show()






# ______________________________________________________________

def main():
    plot()



if __name__ == '__main__':
    main()


