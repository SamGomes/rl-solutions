import numpy
import pandas as pd
import matplotlib.pyplot as plt

import math

class ValueIteration: #estimte the optimal policy
	def __init__(self, numberOfStates, numberOfActions, discountFactor, acceptableEstimationError):

		#a state representesents the current amount of money for each moment
		#an action represents how much the gambler can spend in a bet
		self.numberOfStates = numberOfStates
		self.numberOfActions = numberOfActions

		self.rewardMatrix = numpy.zeros((self.numberOfStates,self.numberOfActions))
		self.rewardMatrix[self.numberOfStates-1] = numpy.ones(self.numberOfActions)	

		self.discountFactor = discountFactor

		self.pMatrices = numpy.zeros((self.numberOfActions, self.numberOfStates, self.numberOfStates));
		self.vMatrix = numpy.zeros(self.numberOfStates);
		
		self.qMatrix = numpy.zeros((self.numberOfStates, self.numberOfActions));
		self.bestActionsForEachState = numpy.zeros(self.numberOfStates);

		self.acceptableEstimationError = acceptableEstimationError

		self.delta = float("inf")

		self.sweep() #compute initial sweep so that the v values are initialized

	def computeProbabilityMatrix(self, action):
		pMatrices = numpy.zeros((self.numberOfActions, self.numberOfStates, self.numberOfStates));
		actionOffset = action + 1 #compensate for the indices starting in 0 and the actions starting in bets of 1 dollar
		if self.pMatrices[action].sum() > 0.0:
			return self.pMatrices[action]
		else:
			for state in range(self.numberOfStates):
				if state > 0 and state < (self.numberOfStates-1):
					if (state < actionOffset or (state + actionOffset) > (self.numberOfStates - 1.0)):
						self.pMatrices[action, state, state] = 1.0 #optional but assumed that the machine refuses the bet and gives back the current money amount
						continue
					else:
						self.pMatrices[action, state, max(state - actionOffset, 0)] = 0.6
						self.pMatrices[action, state, min(state + actionOffset, self.numberOfStates - 1)] = 0.4

			return self.pMatrices[action]


	def computeQMatrixForAction(self, action):
		pMatrixForAction = self.computeProbabilityMatrix(action)
		return self.rewardMatrix[:, action] + self.discountFactor * numpy.dot(pMatrixForAction, self.vMatrix)


	def sweep(self):
		#as we manipulate matrices, all states are updated at once
		self.qMatrix = numpy.zeros((self.numberOfStates, self.numberOfActions))
		oldV = self.vMatrix
		for action in range(self.numberOfActions):
			self.qMatrix[:, action] = self.computeQMatrixForAction(action)

		self.vMatrix = self.qMatrix.max(axis=1)
		self.delta = numpy.linalg.norm(oldV - self.vMatrix)

	def execute(self, numIterations):
		currNumIterations=0

		#value iteration
		while (not ((numIterations!= -1 and currNumIterations >= numIterations) or self.delta < self.acceptableEstimationError)):
			self.sweep()
			currNumIterations = currNumIterations + 1
		self.bestActionsForEachState = self.qMatrix.round(4).argmax(axis=1)


	def getVMatrix(self):
		return self.vMatrix
	
	def getBestActions(self):
		return self.bestActionsForEachState


def main():

	numberOfStates = 101
	numberOfActions = 99
	states = [i for i in range(numberOfStates)]

	# build 1st plot
	plt.figure()
	vIt = ValueIteration(numberOfStates, numberOfActions, 1.0, 1e-8)
	vIt.execute(1) #execute 1st iteration
	vMatrix = vIt.getVMatrix()
	plt.plot(states, vMatrix, color='blue', marker='', linestyle='-', label=r'$iteration\ 1$')
	vIt.execute(1) #execute 2nd iteration
	vMatrix = vIt.getVMatrix()
	plt.plot(states, vMatrix, color='red', marker='', linestyle='-', label=r'$iteration\ 2$')
	vIt.execute(1) #execute 3rd iteration
	vMatrix = vIt.getVMatrix()
	plt.plot(states, vMatrix, color='green', marker='', linestyle='-', label=r'$iteration\ 3$')
	vIt.execute(-1) #execute the rest of iterations
	vMatrix = vIt.getVMatrix()
	plt.plot(states, vMatrix, color='black', marker='', linestyle='-', label=r'$final\ iteration$')
	plt.legend(loc='best')
	plt.ylabel('Value Estimates')
	plt.xlabel('Kept Money')
	plt.show()
	

	# build 2nd plot (draw vertical lines by hand)
	bms = vIt.getBestActions()
	actions = [i for i in range(numberOfActions)]

	plt.figure()
	for i in range(numberOfStates):
		yc = bms[i]
		xc = states[i]

		xc_1 = 0
		yc_1 = 0
		if(i>0):
			xc_1 = states[i-1]
			yc_1 = bms[i-1]

		offset = (xc - xc_1)/2
		# plt.hlines(yc, xc - offset, xc + offset)
		plt.vlines(xc_1 + offset, yc_1, yc)

	plt.plot(states, bms, color='black', marker='_', linestyle='', label=r'$optimal\ policy$')
	plt.legend(loc='best')
	plt.ylabel('Final Policy (Bet amount)')
	plt.xlabel('Kept Money')
	plt.xticks([0,25,50,75,100]) 
	plt.show()

if __name__ == '__main__':
	main()