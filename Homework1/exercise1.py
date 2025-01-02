import numpy
import matplotlib.pyplot as plt

import math

# consider normal distribution for rewards
numOfPossibleActions = 10

trueRewards = []


def bandit(action):
    return numpy.random.normal(trueRewards[action], 1.0)


# numberOfSimSteps=1000 initialReward=0
class ValueBasedAlg:
    def __init__(self, initialReward, numberOfSimSteps, numOfPossibleActions,
                 currRunsRewards, numberOfSims):
        self.numberOfSimSteps = numberOfSimSteps
        self.numOfPossibleActions = numOfPossibleActions
        self.initialReward = initialReward

        self.rewardsPerStep = numpy.zeros(numberOfSimSteps)

        self.reset()

        # chart stuff
        self.currRunsRewards = currRunsRewards
        self.numberOfSims = numberOfSims

    def selectNewAction(self, t):
        pass

    def reset(self):
        self.Q_as = self.initialReward * numpy.ones(self.numOfPossibleActions)
        self.N_as = numpy.zeros(self.numOfPossibleActions)

    def executeSimulation(self):
        for t in range(self.numberOfSimSteps):
            a_t = self.selectNewAction(t)
            r_t = bandit(a_t)
            self.rewardsPerStep[t] += r_t / self.numberOfSims

            self.N_as[a_t] += 1
            n = self.N_as[a_t]

            Q_n = self.Q_as[a_t]
            Q_nPlus1 = Q_n + (1.0 / n * (r_t - Q_n))
            self.Q_as[a_t] = Q_nPlus1

    def getRewardsPerStep(self):
        return self.rewardsPerStep


class EpsilonGreedy(ValueBasedAlg):
    def __init__(self, initialReward, numberOfSimSteps, numOfPossibleActions, currRunsRewards, numberOfSims, epsilon):
        ValueBasedAlg.__init__(self, initialReward, numberOfSimSteps, numOfPossibleActions, currRunsRewards,
                               numberOfSims)
        self.epsilon = epsilon

    def selectNewAction(self, t):
        if (numpy.random.rand() >= self.epsilon):
            # this is to choose a random between values instead of always return the first in a tie break
            Q_asCopy = self.Q_as
            bestQ_asIndexes = numpy.argwhere(Q_asCopy == numpy.amax(Q_asCopy)).flatten().tolist()
            return bestQ_asIndexes[numpy.random.randint(0, len(bestQ_asIndexes))]
        # return 0
        else:
            return numpy.random.randint(0, self.numOfPossibleActions)


class UCB(ValueBasedAlg):
    def __init__(self, initialReward, numberOfSimSteps, numOfPossibleActions, currRunsRewards, numberOfSims, c):
        ValueBasedAlg.__init__(self, initialReward, numberOfSimSteps, numOfPossibleActions, currRunsRewards,
                               numberOfSims)
        self.c = c

    def selectNewAction(self, t):
        ucbs = numpy.zeros(self.numOfPossibleActions)
        for a in range(len(self.Q_as)):
            currEstimate = self.Q_as[a]
            currN = self.N_as[a]
            if (currN == 0):
                # consider a as a maximizing action
                return a
            # calc ucb
            currCB = currEstimate + self.c * numpy.sqrt(numpy.log(t) / self.N_as[a])
            ucbs[a] = currCB

        bestQ_asIndexes = numpy.argwhere(ucbs == numpy.amax(ucbs)).flatten().tolist()

        return bestQ_asIndexes[numpy.random.randint(0, len(bestQ_asIndexes))]


def main():
    global trueRewards

    numberOfSims = 2000
    numberOfStepsPerSim = 1000

    steps = [i for i in range(numberOfStepsPerSim)]

    algorithmsToTest = list()

    currRunAlgRewards = [[0 for t in range(numberOfStepsPerSim)] for a in range(5)]

    algorithmsToTest.append(
        EpsilonGreedy(0, numberOfStepsPerSim, numOfPossibleActions, currRunAlgRewards[0], numberOfSims, 0.0))
    algorithmsToTest.append(
        EpsilonGreedy(5.0, numberOfStepsPerSim, numOfPossibleActions, currRunAlgRewards[1], numberOfSims, 0.0))

    algorithmsToTest.append(
        EpsilonGreedy(0, numberOfStepsPerSim, numOfPossibleActions, currRunAlgRewards[3], numberOfSims, 0.1))
    algorithmsToTest.append(
        EpsilonGreedy(0, numberOfStepsPerSim, numOfPossibleActions, currRunAlgRewards[2], numberOfSims, 0.01))

    algorithmsToTest.append(UCB(0, numberOfStepsPerSim, numOfPossibleActions, currRunAlgRewards[4], numberOfSims, 2.0))

    for i in range(numberOfSims):
        trueRewards = [numpy.random.normal(0.0, 1.0) for a in range(numOfPossibleActions)]

        for algorithmI in range(len(algorithmsToTest)):
            currAlgorithm = algorithmsToTest[algorithmI]
            currAlgorithm.reset()
            currAlgorithm.executeSimulation()

        print('Simulation %d of %d' % ((i + 1), numberOfSims), end='\r')

    for algorithmI in range(len(algorithmsToTest)):
        currRunAlgRewards[algorithmI] = algorithmsToTest[algorithmI].getRewardsPerStep()

    plt.plot(steps, currRunAlgRewards[0], color='green', marker='', linestyle='-', label=r'$\epsilon=0 (greedy)$')
    plt.plot(steps, currRunAlgRewards[1], color='darkviolet', marker='', linestyle='-',
             label=r'$\epsilon=0 (optimistic\ greedy)$')

    plt.plot(steps, currRunAlgRewards[2], color='blue', marker='', linestyle='-', label=r'$\epsilon=0.1$')
    plt.plot(steps, currRunAlgRewards[3], color='red', marker='', linestyle='-', label=r'$\epsilon=0.01$')

    plt.plot(steps, currRunAlgRewards[4], color='orange', marker='', linestyle='-', label=r'$UCB$')

    plt.ylabel('Average Reward')
    plt.xlabel('Steps')

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
