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

		self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

		#return w - self.w_updt
		return  self.w_updt

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
		self.Max = MaxTime # max epocs
		self.TrainData = Train
		self.TestData = Test
		self.NumSamples = Samples

		self.learn_rate  = learnRate
 

		self.minPerf = MinPer
		
		#initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
		np.random.seed() 
		self.W1 = np.random.uniform(-0.5, 0.5, (self.topology[0] , self.topology[1]))  
		#print(self.W1,  ' self.W1')
		self.B1 = np.random.uniform(-0.5,0.5, (1, self.topology[1])  ) # bias first layer
		#print(self.B1, ' self.B1')
		self.BestB1 = self.B1
		self.BestW1 = self.W1 
		self.W2 = np.random.uniform(-0.5, 0.5, (self.topology[1] , self.topology[2]))   
		self.B2 = np.random.uniform(-0.5,0.5, (1,self.topology[2]))  # bias second layer
		self.BestB2 = self.B2
		self.BestW2 = self.W2 
		self.hidout = np.zeros(self.topology[1] ) # output of first hidden layer
		self.out = np.zeros(self.topology[2]) #  output last layer

		self.hid_delta = np.zeros(self.topology[1] ) # output of first hidden layer
		self.out_delta = np.zeros(self.topology[2]) #  output last layer

		adam_learnrate = 0.05

		self.adam_outlayer = Adam(adam_learnrate, 0.9, 0.999)  #learningrate=0.001, b1=0.9, b2=0.999
		self.adam_hidlayer = Adam(adam_learnrate, 0.9, 0.999)  #learningrate=0.001, b1=0.9, b2=0.999

		self.num_layers = len(self.topology)-1


		self.layer = [None]*20  # create list of Layers objects ( just max size of 20 for now - assuming a very deep neural network )

		print(self.topology,  ' topology')

		for n in range(0, self.num_layers): 
			print(self.topology[n],self.topology[n+1], n, ' self.topology[n],self.topology[n+1]')
			self.layer[n] = Layers(self.topology[n],self.topology[n+1], adam_learnrate)
			
		for n in range(0, self.num_layers):  
			print(n, self.layer[n].output)
			print(n, self.layer[n].weights)
 






	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))

	
	def softmax(self, x):
		# Numerically stable with large exponentials
		exps = np.exp(x - x.max())
		return exps / np.sum(exps, axis=0)

	def sampleEr(self,actualout):
		error = np.subtract(self.out, actualout)
		sqerror= np.sum(np.square(error))/self.Top[2] 
		 
		return sqerror

	def ForwardPass(self, X ): 
		z1 = X.dot(self.W1) - self.B1  
		self.hidout = self.sigmoid(z1) # output of first hidden layer   
		z2 = self.hidout.dot(self.W2)  - self.B2 
		self.out = self.sigmoid(z2)  # output second hidden layer

 



	def BackwardPass(self, input_vec, desired):   
		self.out_delta =   (desired - self.out)*(self.out*(1-self.out))  
		self.hid_delta = self.out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout)) #https://www.tutorialspoint.com/numpy/numpy_dot.htm  https://www.geeksforgeeks.org/numpy-dot-python/
  
		self.W2+= self.hidout.T.dot(self.out_delta) * self.learn_rate
		self.B2+=  (-1 * self.learn_rate * self.out_delta)

		self.W1 += (input_vec.T.dot(self.hid_delta) * self.learn_rate) 
		self.B1+=  (-1 * self.learn_rate * self.hid_delta) 
 


	def BackwardPass_Adam(self, input_vec, desired):   

 

		self.out_delta =   (desired - self.out)*(self.out*(1-self.out))  
		self.hid_delta = self.out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))  

		adam_outlayergrad = self.adam_outlayer.update(self.out_delta.copy())
		adam_hidlayergrad = self.adam_hidlayer.update(self.hid_delta.copy())
  
		self.W2+= self.hidout.T.dot(adam_outlayergrad ) 

		self.B2+=  (-1  *  adam_outlayergrad )
 

		self.W1 += input_vec.T.dot(adam_hidlayergrad) 

		self.B1+=  (-1 *  adam_hidlayergrad) 
		 

		#print(self.W1, ' W1')
		#print(self.hid_delta, ' self.hidgrad') 
		#print(adam_hidlayergrad, '  adam_hidlayergrad')

 
 





	 
 
	
	def TestNetwork(self, Data, testSize, tolerance):
		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 
		nOutput = np.zeros((1, self.Top[2]))
		clasPerf = 0
		sse = 0  
		self.W1 = self.BestW1
		self.W2 = self.BestW2 #load best knowledge
		self.B1 = self.BestB1
		self.B2 = self.BestB2 #load best knowledge
 
		for s in range(0, testSize):
							
			Input  =   Data[s,0:self.Top[0]] 
			Desired =  Data[s,self.Top[0]:] 

			self.ForwardPass(Input ) 
			sse = sse+ self.sampleEr(Desired)  


			pred_binary = np.where(self.out > (1 - tolerance), 1, 0)
			
			if( (Desired==pred_binary).all()):
				clasPerf =  clasPerf +1   

			#if(np.isclose(self.out, Desired, atol=erTolerance).any()):
				#clasPerf =  clasPerf +1  

		return ( sse/testSize, float(clasPerf)/testSize * 100 )




	def saveKnowledge(self):
		self.BestW1 = self.W1
		self.BestW2 = self.W2
		self.BestB1 = self.B1
		self.BestB2 = self.B2  

	def BP_GD(self, adam_optimiser):  


		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 
 
  
		Er = [] 
		epoch = 0
		bestmse = 10000 # assign a large number in begining to maintain best (lowest RMSE)
		bestTrain = 0
		while  epoch < self.Max and bestTrain < self.minPerf :
			sse = 0
			for s in range(0, self.NumSamples):
		
				Input[:]  =  self.TrainData[s,0:self.Top[0]]  
				Desired[:]  = self.TrainData[s,self.Top[0]:]  

				self.ForwardPass(Input)  

				if adam_optimiser == True:
					self.BackwardPass_Adam(Input ,Desired)
				else:
					self.BackwardPass(Input ,Desired)

				sse = sse+ self.sampleEr(Desired)
			 
			mse = np.sqrt(sse/self.NumSamples*self.Top[2])

			if mse < bestmse:
				 bestmse = mse
				 #print(bestmse, epoch, ' bestmse epoch')
				 self.saveKnowledge() 
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
		MaxTime = 20 #500


		 

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
 


	Topo = [Input, Hidden, Output] 
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

