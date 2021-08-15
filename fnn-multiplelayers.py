# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#https://github.com/rohitash-chandra
 

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  
 

#Sigmoid units used in hidden and output  

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
 

 

import matplotlib.pyplot as plt
import numpy as np 
import random
import time


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

	def __init__(self, topology, Train, Test, MaxTime, Samples, MinPer, learnRate): 
		self.topology  = topology  # NN topology [input, hidden, output]

		self.output_activationfunc = 'sigmoid'


		self.Top  = topology  # NN topology [input, hidden, output]


		self.Max = MaxTime # max epocs
		self.TrainData = Train
		self.TestData = Test
		self.NumSamples = Samples

		self.sgdlearn_rate  = learnRate
 

		self.minPerf = MinPer
		 
		np.random.seed()   

		self.adam_learnrate = 0.05

		self.adam_outlayer = Adam(self.adam_learnrate, 0.9, 0.999)  #learningrate=0.001, b1=0.9, b2=0.999
		self.adam_hidlayer = Adam(self.adam_learnrate, 0.9, 0.999)  #learningrate=0.001, b1=0.9, b2=0.999

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
		else: # linear output activation
			self.layer[last_index].gradient =   (desired - self.layer[last_index].output)*(self.layer[last_index].output*(1-self.layer[last_index].output))  


		for n in range(self.end_index-1,0, -1):   
			self.layer[n].gradient = self.layer[n+1].gradient.dot(self.layer[n+1].weights.T) * (self.layer[n].output * (1-self.layer[n].output))  
		
		for n in range(self.end_index,0, -1):  
		 
			self.layer[n].weights += self.layer[n-1].output.T.dot(self.layer[n].gradient) * self.sgdlearn_rate
			self.layer[n].bias  +=  -1 * self.sgdlearn_rate * self.layer[n].gradient
 
 
 


	def BackwardPass_Adam(self, input_vec, desired):   

 

		self.out_delta =   (desired - self.out)*(self.out*(1-self.out))  
		self.hid_delta = self.out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))  

		adam_outlayergrad = self.adam_outlayer.update(self.out_delta.copy())
		adam_hidlayergrad = self.adam_hidlayer.update(self.hid_delta.copy())
  
		self.W2+= self.hidout.T.dot(adam_outlayergrad ) 

		self.B2+=  (-1  *  adam_outlayergrad )
 
		self.W1 += input_vec.T.dot(adam_hidlayergrad) 

		self.B1+=  (-1 *  adam_hidlayergrad) 
		 
 
	
	def TestNetwork(self, data_features, testSize, tolerance):


		#Input = np.zeros((1, self.Top[0])) # temp hold input
		desired = np.zeros((1, self.topology[self.end_index])) 
 
		#nOutput = np.zeros((1, self.topology[self.end_index]))

		clasPerf = 0
		sse = 0  
		'''self.W1 = self.BestW1
		self.W2 = self.BestW2 #load best knowledge
		self.B1 = self.BestB1
		self.B2 = self.BestB2 #load best knowledge'''
 
		for s in range(0, testSize):
							
			features  =   data_features[s,0:self.Top[0]] 
			desired =  data_features[s,self.Top[0]:] 

			self.forward_pass(features) 
			sse = sse+ self.individual_error(desired)  


			pred_binary = np.where(self.layer[self.end_index].output > (1 - tolerance), 1, 0)
			
			if( (desired==pred_binary).all()):
				clasPerf =  clasPerf +1    

		return ( sse/testSize, float(clasPerf)/testSize * 100 )




	def saveKnowledge(self):
		self.BestW1 = self.W1
		self.BestW2 = self.W2
		self.BestB1 = self.B1
		self.BestB2 = self.B2  

	def BP_GD(self, adam_optimiser):  


		Input = np.zeros((1, self.topology[0])) # temp hold input
		Desired = np.zeros((1, self.topology[self.end_index])) 
 
  
		Er = [] 
		epoch = 0
		bestmse = 10000 # assign a large number in begining to maintain best (lowest RMSE)
		bestTrain = 0
		while  epoch < self.Max and bestTrain < self.minPerf :
			sse = 0
			for s in range(0, self.NumSamples):
		
				Input[:]  =  self.TrainData[s,0:self.topology[0]]  
 
				Desired[:]  = self.TrainData[s,self.topology[0]:]  

				#self.ForwardPass(Input)  

				data_features = Input
 
				self.forward_pass(data_features)  

				if adam_optimiser == True:
					self.backward_pass(Input ,Desired)
				else:
					self.backward_pass(Input ,Desired)

				sse = sse+ self.individual_error(Desired)
			 
			mse = np.sqrt(sse/self.NumSamples*self.topology[self.end_index])

			if mse < bestmse:
				 bestmse = mse 
				 #self.saveKnowledge() 
				 (x,bestTrain) = self.TestNetwork(self.TrainData, self.NumSamples, 0.2)

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

def main(): 
					
		
	problem = 1 # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)
				

	if problem == 1:
		TrDat  = np.loadtxt("data/train.csv", delimiter=',') #  Iris classification problem (UCI dataset)
		TesDat  = np.loadtxt("data/test.csv", delimiter=',') #  
		Hidden = 6
		Input = 4
		Output = 2 #https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
		TrSamples =  TrDat.shape[0]
		TestSize = TesDat.shape[0]
		learnRate = 0.1  
		TrainData  = normalisedata(TrDat, Input, Output) 
		TestData  = normalisedata(TesDat, Input, Output)
		MaxTime = 500 #500


		 

	elif problem == 2:
		TrainData = np.loadtxt("data/4bit.csv", delimiter=',') #  4-bit parity problem
		TestData = np.loadtxt("data/4bit.csv", delimiter=',') #  
		Hidden = 6
		Input = 4
		Output = 1 #  https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
		TrSamples =  TrainData.shape[0]
		TestSize = TestData.shape[0]
		learnRate = 0.9 
		MaxTime = 200 #1000

	elif problem == 3:
		TrainData = np.loadtxt("data/xor.csv", delimiter=',') #  XOR  problem
		TestData = np.loadtxt("data/xor.csv", delimiter=',') #  
		Hidden = 3
		Input = 2
		Output = 2  # one hot encoding: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
		TrSamples =  TrainData.shape[0]
		TestSize = TestData.shape[0]
		learnRate = 0.5
		MaxTime = 500
 


	Topo = [Input,  Hidden, Hidden-2, Hidden, Output] 
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

	adam_optimiser = True # False means you use SGD 



	for run in range(0, MaxRun  ):
		print(run, ' is experimental run') 

		fnn = Network(Topo, TrainData, TestData, MaxTime, TrSamples, MinCriteria, learnRate)
		start_time=time.time()
		(erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnn.BP_GD(adam_optimiser)   

		Time[run]  =time.time()-start_time
		(testMSE[run], testPerf[run]) = fnn.TestNetwork(TestData, TestSize, testTolerance)
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

