# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#https://github.com/rohitash-chandra
 

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  
 

#Sigmoid units used in hidden and output  

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
 

 

import matplotlib.pyplot as plt
import numpy as np 
import random
import time


from numpy import *


# this is to compare results with sk-learn or to process data 
from sklearn import datasets 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier



class Adam():
	def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
		self.learning_rate = learning_rate
		self.eps = 1e-8
		self.m = None
		self.v = None
		# Decay rates
		self.b1 = b1
		self.b2 = b2

	#def update(self, w, grad_wrt_w):
	
	def update(self, grad_wrt_w):
		# If not initialized
		if self.m is None:
			self.m = np.zeros(np.shape(grad_wrt_w))
			self.v = np.zeros(np.shape(grad_wrt_w))
		
		self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
		self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

		m_hat = self.m / (1 - self.b1)
		v_hat = self.v / (1 - self.b2)

		self.grad = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

		#return w - self.w_updt
		return  self.grad

class Layers:

	def __init__(self, first, second,adam_learnrate):  
		#self.number = first

		self.weights = np.random.uniform(-0.5, 0.5, (first , second))   
		self.bias = np.random.uniform(-0.5,0.5, (1, second))  # bias second layer

		self.output = np.zeros(second) # output of  layer 
		self.gradient = np.zeros(second) # gradient of layer 


		self.adam_opt = Adam(adam_learnrate, 0.9, 0.999)  #learningrate=0.001, b1=0.9, b2=0.999

		

 
class Network:

	def __init__(self, topology, x_train, x_test, y_train, y_test, max_epocs, num_samples, min_error, learn_rate): 
		self.topology  = topology  # NN topology [input, hidden, output]

		self.output_activationfunc = 'sigmoid'


		#self.Top  = topology  # NN topology [input, hidden, output]


		self.max_epocs = max_epocs # max epocs
		#self.TrainData = Train
		#self.TestData = Test

		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test


		self.num_samples = num_samples

		self.sgdlearn_rate  = learn_rate
 

		self.min_error = min_error 
		 
		np.random.seed()   

		self.adam_learnrate = 0.05
 

		self.end_index = len(self.topology)-1


		self.layer = [None]*20  # create list of Layers objects ( just max size of 20 for now - assuming a very deep neural network )

		print(self.topology,  ' topology')

		self.layer[0] = Layers(1,1, 0) # this is just for layer where input features are stored

		for n in range(1,  len(self.topology)):  
			self.layer[n] = Layers(self.topology[n-1],self.topology[n], self.adam_learnrate)
			
		#for n in range(0, self.num_layers):  
		#	print(n, self.layer[n].output)
		#	print(n, self.layer[n].weights)
 
 

	def activation(self,x):
		if  self.output_activationfunc == 'sigmoid':
			return 1 / (1 + np.exp(-x))
		elif self.output_activationfunc == 'step': # todo
			if x > 0:
				return 1  #todo
			else:
				return 0

		else:
			return x # linear acivation

	
	def softmax(self, x):
		# Numerically stable with large exponentials
		exps = np.exp(x - x.max())
		return exps / np.sum(exps, axis=0)

	def individual_error(self,desired):
		error = np.subtract(self.layer[self.end_index].output, desired)
		sq_error= np.sum(np.square(error))/self.topology[self.end_index] 
		return sq_error

	def forward_pass(self, input_features ): 

		self.layer[0].output = input_features 

		for n in range(0, self.end_index):  
			z = self.layer[n].output.dot(self.layer[n+1].weights) - self.layer[n+1].bias 
			self.layer[n+1].output = self.activation(z) 

	def backward_pass(self, input_features, desired):   

		last_index= self.end_index
		if self.output_activationfunc == 'sigmoid': #sigmoid
			self.layer[last_index].gradient =   (desired - self.layer[last_index].output)*(self.layer[last_index].output*(1-self.layer[last_index].output))
		else: # linear or step output activation
			self.layer[last_index].gradient =   (desired - self.layer[last_index].output) * self.layer[last_index].output 


		for n in range(self.end_index-1,0, -1):   
			self.layer[n].gradient = self.layer[n+1].gradient.dot(self.layer[n+1].weights.T) * (self.layer[n].output * (1-self.layer[n].output))  
		
		for n in range(self.end_index,0, -1):  
		 
			self.layer[n].weights += self.layer[n-1].output.T.dot(self.layer[n].gradient) * self.sgdlearn_rate
			self.layer[n].bias  +=  -1 * self.sgdlearn_rate * self.layer[n].gradient
 

	def backwardpass_advancedgradients(self, input_vec, desired):   

		last_index= self.end_index
		if self.output_activationfunc == 'sigmoid': #sigmoid
			self.layer[last_index].gradient =   (desired - self.layer[last_index].output)*(self.layer[last_index].output*(1-self.layer[last_index].output))
		else: # linear or step output activation
			self.layer[last_index].gradient =   (desired - self.layer[last_index].output) * self.layer[last_index].output 

		self.layer[last_index].gradient_adv = self.layer[last_index].adam_opt.update(self.layer[last_index].gradient)


		for n in range(self.end_index-1,0, -1):   
			self.layer[n].gradient = self.layer[n+1].gradient.dot(self.layer[n+1].weights.T) * (self.layer[n].output * (1-self.layer[n].output)) 
			self.layer[n].gradient_adv = self.layer[n].adam_opt.update(self.layer[n].gradient) 
		
		for n in range(self.end_index,0, -1):  
		 
			self.layer[n].weights += self.layer[n-1].output.T.dot(self.layer[n].gradient_adv)  
			self.layer[n].bias  +=  -1  * self.layer[n].gradient_adv
 
		 
 
	
	def test_network(self, features_x, desired_x, tolerance):
 
		desired = np.zeros((1, self.topology[self.end_index])) 

		size = features_x.shape[1]
  

		classification = 0
		sse = 0  
		'''self.W1 = self.BestW1
		self.W2 = self.BestW2 #load best knowledge
		self.B1 = self.BestB1
		self.B2 = self.BestB2 #load best knowledge'''
 
		for s in range(0, size):
							
			features  =   features_x[s,:] 
			desired =  desired_x[s,:] 

			self.forward_pass(features) 
			sse = sse+ self.individual_error(desired)  


			pred_binary = np.where(self.layer[self.end_index].output > (1 - tolerance), 1, 0)
			
			if( (desired==pred_binary).all()):
				classification =  classification +1    

		return ( sse/size, float(classification)/size * 100 )




	def saveKnowledge(self):
		#todo
		'''self.BestW1 = self.W1
		self.BestW2 = self.W2
		self.BestB1 = self.B1
		self.BestB2 = self.B2  '''

	def backpropagation(self, optimiser):  


		data_features = np.zeros((1, self.topology[0])) # temp hold input
		desired = np.zeros((1, self.topology[self.end_index])) 
 
  
		Er = [] 
		epoch = 0
		bestmse = 10000 # assign a large number in begining to maintain best (lowest RMSE)
		best_train = 0
		while  epoch < self.max_epocs and best_train < self.min_error :
			sse = 0
			for s in range(0, self.num_samples):
		
				#data_features[:]  =  self.TrainData[s,0:self.topology[0]]  

				data_features[:]  =  self.x_train[s,:]  
 
				desired[:]  = self.y_train[s,:] 
 
 
				self.forward_pass(data_features)  

				if optimiser == 'adam':
					self.backwardpass_advancedgradients(data_features ,desired)
				else: 
					self.backward_pass(data_features ,desired)

				sse = sse+ self.individual_error(desired)
			 
			mse = np.sqrt(sse/self.num_samples*self.topology[self.end_index])

			if mse < bestmse:
				 bestmse = mse 
				 #self.saveKnowledge() 
				 (x,bestTrain) = self.test_network(self.x_train, self.y_train,  0.2)

			Er = np.append(Er, mse)
			
			epoch=epoch+1  


		
		#print(self.BestW1, 'W1')
		#print(self.BestW2, ' W2')

		return (Er,bestmse, bestTrain, epoch) 



def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]
	traindt = data[:,np.array(range(0,inputsize))]  
	dt = np.amax(traindt, axis=0)
	tds = abs(traindt/dt) 
	return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)



def read_data(problem):

	if problem ==1:
		#Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
		data_in = genfromtxt("data/pima-indians-diabetes.csv", delimiter=",")
		data_inputx = data_in[:,0:8] # all raw features 0, 1, 2, 3, 4, 5, 6, 7 
		transformer = Normalizer().fit(data_inputx)  # fit does nothing. (scikit learn)
		data_inputx = transformer.transform(data_inputx)

		data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

		# needs one hot encoding 
	elif problem ==2:
		#Iris with one hot encoded labels
		data_in = genfromtxt("data/iris.csv", delimiter=",")
		data_inputx = data_in[:,0:4] # all raw features [0, 1, 2, 3]
		transformer = Normalizer().fit(data_inputx)  # fit does nothing. (scikit learn)
		data_inputx = transformer.transform(data_inputx)

		#data_inputx = transformer.transform(data_inputx)
		data_inputy = data_in[:,4:7] # one hot encded labels (3 classes) [4,5,6]


	elif problem ==3: 
		#wine
		data_in = genfromtxt("data/wine.csv", delimiter=",")
		data_inputx = data_in[:,0:13] # all features [0, 1, 2, 3, ... 12] (13 raw features )
		#data_inputx = transformer.transform(data_inputx)
		data_inputy = data_in[:,-1] # 
		

		# needs one hot encoding 

	else:
		print('else')






	x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=1)

	return x_train, x_test, y_train, y_test

def main(): 

	#assume you use one hot encoding always in labels for classification problems
					
		
	problem = 2 # iris works, rest need one hot encoding fixed

	x_train, x_test, y_train, y_test  = read_data(problem)

	#print(x_train, y_train, ' x_train,  y_train ')

	input_features = x_train.shape[1]
	num_outputs =  y_train.shape[1]

	TrSamples =  x_train.shape[0]
	TestSize = x_test.shape[0]

	learnRate = 0.1  

	hidden = 5 # let user decide for prob
 
 


	Topo = [input_features, hidden,  hidden, num_outputs] 

	MaxTime = 500 # epochs


	MaxRun = 1 # number of experimental runs 
	 
	MinCriteria = 95 #stop when learn 95 percent

	trainTolerance = 0.2 # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
	testTolerance = 0.49
 

	trainPerf = np.zeros(MaxRun)
	testPerf =  np.zeros(MaxRun)

	trainMSE =  np.zeros(MaxRun)
	testMSE =  np.zeros(MaxRun)
	Epochs =  np.zeros(MaxRun)
	Time =  np.zeros(MaxRun)

	optimiser = 'adam' # 'sgd' 



	for run in range(0, MaxRun  ):
		print(run, ' is experimental run') 

		fnn = Network(Topo, x_train, x_test, y_train, y_test, MaxTime, TrSamples, MinCriteria, learnRate)
		start_time=time.time()
		(erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnn.backpropagation(optimiser)   

		Time[run]  =time.time()-start_time
		(testMSE[run], testPerf[run]) = fnn.test_network(x_test, y_test,  testTolerance)


	print(' print classification performance for each experimental run') 
	print(trainPerf)
	print(testPerf)
	print(' print RMSE performance for each experimental run') 
	print(trainMSE)
	print(testMSE)
	print(' print Epocs and Time taken for each experimental run') 
	print(Epochs)
	print(Time)
	print(' print mean and std of training performance') 
	

	print(np.mean(trainPerf), np.std(trainPerf))
	print(np.mean(testPerf), np.std(testPerf))

	print(' print mean and std of computational time taken') 
	
	print(np.mean(Time), np.std(Time))
	
	
				 
	plt.figure()
	plt.plot(erEp )
	plt.ylabel('error')  
	plt.savefig('out.png')
			 
 
if __name__ == "__main__": main()

