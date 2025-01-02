import numpy
import matplotlib.pyplot as plt
import math
import pdb

#---------------- Problem Globals ------------------------

numberOfStates = 7
numberOfActionsPerState = 2


def approximateQ(state, action, actionsFeatures, wVector):
    return numpy.dot(actionsFeatures[action][state], wVector)


def getNextState(currState, currAction):
    if currAction == 0:
        return 6
    if currAction == 1:
        return int(numpy.floor(numpy.random.rand() * 5))


#---------------------------------------------------------


class EpsilonAB():
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def selectNewAction(self, currState, possibleActions, actionsFeatures, wVector):
        if numpy.random.rand() >= self.epsilon:
            return possibleActions[0]
        else:
            return possibleActions[1]


class Greedy():
    def selectNewAction(self, currState, possibleActions, actionsFeatures, wVector):
        bestQ = approximateQ(currState, possibleActions[0], actionsFeatures, wVector)
        bestAction = possibleActions[0]
        for i in range(len(possibleActions) - 1):
            currQ = approximateQ(currState, possibleActions[i + 1], actionsFeatures, wVector)
            if (currQ > bestQ):
                bestQ = currQ
                bestAction = possibleActions[i + 1]
        return bestAction


class LinearApproximationLearningAlg:
    def __init__(self, numberOfStates, numberOfActionsPerState, initialState, \
                 discountFactor, stepSize, rewardMatrix, targetPolicy, behaviorPolicy, \
                 actionsFeatures, initialWVector):

        self.initialState = initialState
        self.actionsFeatures = actionsFeatures
        self.numberOfStates = numberOfStates
        self.numberOfActionsPerState = numberOfActionsPerState
        self.rewardMatrix = rewardMatrix
        self.discountFactor = discountFactor
        self.wVector = initialWVector
        self.stepSize = stepSize
        self.targetPolicy = targetPolicy
        self.behaviorPolicy = behaviorPolicy

        self.currPolicy = numpy.array([])
        self.possibleActions = numpy.arange(self.numberOfActionsPerState)
        self.currState = self.initialState
        self.currAction = self.behaviorPolicy.selectNewAction(self.currState, self.possibleActions,
                                                              self.actionsFeatures, self.wVector)

    def iterate(self):
        nextState = getNextState(self.currState, self.currAction)
        currReward = self.rewardMatrix[self.currState, self.currAction]

        nextAction = self.behaviorPolicy.selectNewAction(nextState, self.possibleActions, self.actionsFeatures,
                                                         self.wVector)
        if (self.behaviorPolicy == self.targetPolicy):
            qAction = nextAction
        else:
            qAction = self.targetPolicy.selectNewAction(nextState, self.possibleActions, self.actionsFeatures,
                                                        self.wVector)

        currentQValue = approximateQ(self.currState, self.currAction, self.actionsFeatures, self.wVector)
        futureQValue = approximateQ(nextState, qAction, self.actionsFeatures, self.wVector)

        self.wVector = self.wVector + self.stepSize * self.actionsFeatures[self.currAction][self.currState] * (
                    currReward + self.discountFactor * futureQValue - currentQValue)
        self.currPolicy = numpy.append(self.currPolicy, self.currAction)

        self.currState = nextState
        self.currAction = nextAction

    def getCurrPolicy(self):
        return self.currPolicy

    def getWVector(self):
        return self.wVector


def main():
    # init algorithm parameters
    numberOfRuns = 1000
    numberOfStepsPerRun = 500
    rewardMatrix = numpy.zeros((numberOfStates, numberOfActionsPerState))  #unefficient but allows modifications
    discountFactor = 0.99
    stepSize = 0.01
    featureCount = 15

    # init feature matrices
    halfIndex = int(numpy.floor(featureCount / 2))
    fiA = numpy.zeros((numberOfStates, featureCount))
    fiA[:, halfIndex] = 1
    for i in range(halfIndex):
        fiA[i, i] = 2
    fiA[numberOfStates - 1, halfIndex] = 2
    fiA[numberOfStates - 1, halfIndex - 1] = 1

    fiB = numpy.zeros((numberOfStates, featureCount))
    for i in range(halfIndex):
        fiB[i, halfIndex + i + 1] = 1

    actionsFeatures = []
    actionsFeatures.append(fiA)
    actionsFeatures.append(fiB)

    initialWVector = numpy.full(featureCount, 1)
    initialWVector[halfIndex - 1] = 10

    # init charts stuff
    steps = [i for i in range(numberOfStepsPerRun)]
    sarsaEpisodesNorms = numpy.zeros(numberOfStepsPerRun)
    qLEpisodesNorms = numpy.zeros(numberOfStepsPerRun)

    # init and iterate both algorithms numRuns times
    epsilonAB = EpsilonAB(6.0 / 7.0)
    greedy = Greedy()
    for r in range(numberOfRuns):
        qLearning = LinearApproximationLearningAlg(numberOfStates, numberOfActionsPerState, 0, discountFactor, stepSize,
                                                   rewardMatrix, greedy, epsilonAB, actionsFeatures, initialWVector)
        sarsa = LinearApproximationLearningAlg(numberOfStates, numberOfActionsPerState, 0, discountFactor, stepSize,
                                               rewardMatrix, epsilonAB, epsilonAB, actionsFeatures, initialWVector)

        for i in range(numberOfStepsPerRun):
            print("Run " + str(r) + " of " + str(numberOfRuns), end='\r')
            sarsa.iterate();
            qLearning.iterate();

            currSARSANorm = numpy.linalg.norm(sarsa.getWVector());
            currQLNorm = numpy.linalg.norm(qLearning.getWVector());

            sarsaEpisodesNorms[i] += currSARSANorm / numberOfRuns
            qLEpisodesNorms[i] += currQLNorm / numberOfRuns

    # plot data
    plt.plot(steps, sarsaEpisodesNorms, label=r'$SARSA$', color="blue")
    plt.plot(steps, qLEpisodesNorms, label=r'$Q-Learning$', color="red")

    plt.legend(loc='best')
    plt.xlabel('Step')
    plt.ylabel('Weight Vector Norm')
    plt.show()


if __name__ == '__main__':
    main()
