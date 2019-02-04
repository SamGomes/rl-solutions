import numpy
import matplotlib.pyplot as plt
import enum
import math
from scipy.interpolate import interp1d

cliffWidth = 12;
cliffHeight = 4;


#---------------cliff problem modeling methods-------------------
def getCliffIndex(line,column):
	return column*cliffWidth + line
def getCliffLineColumn(index):
	column = index // cliffWidth;
	line = index - cliffWidth * column
	return numpy.array([line,column])

def nextCliffState(currState, initialState, action):
	lineColumn = getCliffLineColumn(currState)
	line = lineColumn[0]
	column = lineColumn[1]

	switcher = {
		0: getCliffIndex(line-1,column),
		1: getCliffIndex(line,column-1),
		2: getCliffIndex(line+1,column),
		3: getCliffIndex(line,column+1)
	}
	return switcher.get(action, "Invalid action: "+ str(action))

def hasFalledCliff(state): 
	indexPos = getCliffLineColumn(state);
	line = indexPos[0]
	column = indexPos[1]
	return (column == cliffHeight-1 and line > 0 and line < cliffWidth-1)		

def possibleNextCliffActions(index):
	indexPos = getCliffLineColumn(index);
	line = indexPos[0]
	column = indexPos[1]
	nextActions = numpy.array([], dtype=int)
	
	if(line>0):
		nextActions = numpy.append(nextActions, 0) # left 
	if(column>0):
		nextActions = numpy.append(nextActions, 1) # up
	if(line<cliffWidth-1):
		nextActions = numpy.append(nextActions, 2) # right 
	if(column<cliffHeight-1):
		nextActions = numpy.append(nextActions, 3) # down

	return nextActions;


class EpsilonGreedy(): 
	def __init__(self, epsilon):
		self.epsilon = epsilon

	def selectNewAction(self, Q, currState, possibleActions):
		if (numpy.random.rand() >= self.epsilon):
			# select greedy action 

			# the list of maximum values is considered to choose a random between values instead of always return the first in a tie break
			stateQs = Q[currState, possibleActions]
			bestQIndexes = numpy.argwhere(stateQs == numpy.max(stateQs)).flatten().tolist()
			return possibleActions[numpy.random.choice(bestQIndexes)]
		else:
			# select random action
			return numpy.random.choice(possibleActions)



class TabularLearningAlg:
	def __init__(self, numberOfStates, numberOfActionsPerState, initialState, terminalState, \
				discountFactor, stepSize, rewardMatrix, targetPolicy, behaviorPolicy):

		self.initialState = initialState
		self.terminalState = terminalState

		self.numberOfStates = numberOfStates
		self.numberOfActionsPerState = numberOfActionsPerState

		self.rewardMatrix = rewardMatrix	
		self.qMatrix = numpy.random.rand(self.numberOfStates, self.numberOfActionsPerState);

		self.discountFactor = discountFactor
		self.stepSize = stepSize

		self.targetPolicy = targetPolicy;
		self.behaviorPolicy = behaviorPolicy;

		self.totalEpisodeReward = 0
		self.currPolicy = numpy.array([])

	def iterate(self):

		self.totalEpisodeReward = 0

		currState = self.initialState;
		currAction = self.behaviorPolicy.selectNewAction(self.qMatrix, currState, possibleNextCliffActions(currState))

		while(1):
			self.currPolicy = numpy.append(self.currPolicy, currState)

			# Take action A, observe R, S'
			nextState = nextCliffState(currState, self.initialState, currAction)
			currReward = self.rewardMatrix[currState]
			self.totalEpisodeReward += currReward

			fellOfCliff = hasFalledCliff(currState)
			if fellOfCliff:
				nextState = self.initialState;

			# Choose A' from S' using policy derived from Q
			nextAction = self.behaviorPolicy.selectNewAction(self.qMatrix, nextState, possibleNextCliffActions(nextState))
			if(self.behaviorPolicy==self.targetPolicy):
				qAction = nextAction
			else:
				qAction = self.targetPolicy.selectNewAction(self.qMatrix, nextState, possibleNextCliffActions(nextState))
			
			# update qMatrix
			currentQValue = self.qMatrix[currState, currAction]
			if currState == self.terminalState  or fellOfCliff:
				futureQValue = 0
			else:
				futureQValue = self.qMatrix[nextState, qAction]

			self.qMatrix[currState, currAction] = currentQValue + self.stepSize * (currReward + self.discountFactor * futureQValue - currentQValue)

			if currState == self.terminalState or fellOfCliff:
				return

			currState = nextState
			currAction = nextAction

	def getTotalEpisodeReward(self):
		return self.totalEpisodeReward
			
	def getQMatrix(self):
		return self.qMatrix


	def getCurrPolicy(self):
		return self.currPolicy


def displayPath(ax, avgQs, initialState, terminalState, gridX, gridY, arrowColor, arrowOffsetX, arrowOffsetY, arrowSize):

	greedyStrategy = EpsilonGreedy(0)
	currState = initialState
	currStatePos = getCliffLineColumn(currState)

	while(currState != terminalState):
		currAction = greedyStrategy.selectNewAction(avgQs, currState, possibleNextCliffActions(currState))
		nextState =  nextCliffState(currState, initialState, currAction)
		nextStatePos = getCliffLineColumn(nextState)

		orientation = nextStatePos - currStatePos

		directionalArrowOffset = (1+orientation)/4 + [arrowOffsetX,arrowOffsetY]
		directionalArrowOffset =  [arrowOffsetX,arrowOffsetY]

		ax.arrow(currStatePos[0]-directionalArrowOffset[0], currStatePos[1]-directionalArrowOffset[1], orientation[0]*arrowSize, orientation[1]*arrowSize, head_width=0.05, head_length=0.1,  ec=arrowColor, color=arrowColor)

		currState = nextState
		currStatePos = nextStatePos


def main():

	# set parameters acording to the problem
	numberOfRuns = 1000

	numberOfStates = cliffWidth*cliffHeight
	numberOfActionsPerState = 4
	initialState = getCliffIndex(0,3)
	terminalState = getCliffIndex(cliffWidth-1,cliffHeight-1);
	discountFactor = 1.0
	stepSize = 0.5

	numberOfEpisodes = 500
	
	rewardMatrix = numpy.full(numberOfStates, -1)
	rewardMatrix[getCliffIndex(1,cliffHeight-1):getCliffIndex(cliffWidth-1,cliffHeight-1)] = -100
	

	# define chart variables
	episodes = [i for i in range(numberOfEpisodes)]

	sarsaEpisodesRSums = numpy.zeros(numberOfEpisodes)
	eGreedy = EpsilonGreedy(0.15)
	
	qLEpisodesRSums = numpy.zeros(numberOfEpisodes)
	greedy = EpsilonGreedy(0)

	gridX = numpy.arange(cliffWidth)
	gridY = numpy.arange(cliffHeight)
	gridPolicyFrequenciesSarsa = numpy.zeros((cliffHeight,cliffWidth))
	gridPolicyFrequenciesQL = numpy.zeros((cliffHeight,cliffWidth))

	sarsaAvgQs = numpy.zeros((numberOfStates, numberOfActionsPerState))
	qLearningAvgQs = numpy.zeros((numberOfStates, numberOfActionsPerState))

	avgNumCliffFallsSARSA = numpy.zeros(numberOfEpisodes)
	avgNumCliffFallsQ = numpy.zeros(numberOfEpisodes)


	# execute simulation runs
	for r in range(numberOfRuns):
		sarsa = TabularLearningAlg(numberOfStates, numberOfActionsPerState, initialState, terminalState, discountFactor, stepSize, rewardMatrix, eGreedy, eGreedy)
		qLearning = TabularLearningAlg(numberOfStates, numberOfActionsPerState, initialState, terminalState, discountFactor, stepSize, rewardMatrix, greedy, eGreedy)
		print("Run "+ str(r) +" of " + str(numberOfRuns), end='\r')
		for i in range(numberOfEpisodes):
			sarsa.iterate();
			qLearning.iterate();

			currSARSASum = sarsa.getTotalEpisodeReward() /numberOfRuns;
			currQLSum = qLearning.getTotalEpisodeReward() /numberOfRuns;

			sarsaEpisodesRSums[i] += currSARSASum;
			qLEpisodesRSums[i] += currQLSum;

		sarsaAvgQs += sarsa.getQMatrix()/numberOfRuns
		qLearningAvgQs += qLearning.getQMatrix()/numberOfRuns




	# smooth reward sum lines and plot them
	splineSARSA = interp1d(episodes, sarsaEpisodesRSums)
	splineQ = interp1d(episodes, qLEpisodesRSums)
	plt.plot(episodes, splineSARSA(episodes), label=r'$SARSA$', color = "blue")
	plt.plot(episodes, splineQ(episodes), label=r'$Q-Learning$', color = "red")

	plt.legend(loc='best')
	plt.xlabel('Episode')
	plt.ylabel('Rewards Sum')
	plt.show()



	# display policy

	# draw grid
	plt.xticks([]) 
	plt.yticks([]) 
	ax = plt.axes()

	x = numpy.array([1, cliffWidth-1 ,cliffWidth-1, 1])
	y = numpy.array([0, 0, 1, 1])
	plt.fill((cliffWidth-x)-0.5,(cliffHeight-y)-0.5, c='grey', alpha=0.3)
	plt.text(cliffWidth/2 - 3, cliffHeight - 0.85, "T h e  C l i f f", fontsize=15)

	for i in range(len(gridX)+1):
		ax.plot([i-0.5, i-0.5], [-0.5,cliffHeight-0.5], color="grey") 
	
	for j in range(len(gridY)+1):
		ax.plot([0-0.5, cliffWidth-0.5], [j-0.5, j-0.5], color="grey") 


	displayPath(ax, sarsaAvgQs, initialState, terminalState, gridX, gridY, "blue", 0.3, 0.0, 0.3)
	displayPath(ax, qLearningAvgQs, initialState, terminalState, gridX, gridY, "red", 0.1, 0.0, 0.3)

	plt.gca().invert_yaxis()
	plt.show()



if __name__ == '__main__':
	main()
