# For matrix operations as all input, output and weights are in matrix form
import numpy as np

# Dataset
train_data = np.array([[1,0,1],[1,1,1],[0,1,0],[0,0,0]])                                                                   
train_labels = np.array([[1,1,1,0]]).T                                                                                      
test_data = np.array([[1,1,0],[1,0,0]])                                                                                       

# Class of a neural network of 2 hidden layers
class nn():
	# initializing weights for both hidden layers
	def __init__(self):
		self.__layer1_weights = 2*(np.random.random(train_data.shape).T)-1                              
		self.__layer2_weights = 2*np.random.random(train_labels.shape)-1                                
	
	# activating function of the perceptron
	def activate(self,x,deriv=False):
		if(deriv==True):
			return x*(1-x)
		return 1/(1+np.exp(-x))

	# training our neural network
	def train(self ,x,y):
		for i in range(60000):                                                                                            # You can change number of iterations according to dataset
			# initializing all layers
			input_layer = x                                                                                                 # Input Layer
			h_layer1 = self.activate(np.dot(x,self.__layer1_weights))                                # 1st hidden layer
			h_layer2 = self.activate(np.dot(h_layer1,self.__layer2_weights))                    # 2nd hidden layer
			output_layer = h_layer2                                                                                   # Output Layer 

			# Calculating errors and gradients for both hidden layers
			e_layer2 = y - output_layer                                                                               # Error in hidden layer 2
			g_layer2 = e_layer2*self.activate(h_layer2,deriv=True)                                    # Gradient to minimize error in hidden layer 2
			e_layer1 = g_layer2.dot((self.__layer2_weights).T)                                          # Error in hidden layer 1
			g_layer1 = e_layer1*self.activate(h_layer1,deriv = True)                                  #  Gradient to minimize error in hidden layer 1

			# Updating our original weights using gradients
			self.__layer2_weights += h_layer1.T.dot(g_layer2)                                    
			self.__layer1_weights += input_layer.T.dot(g_layer1)                                   

	# testing of our neural network
	def test(self,x):
		input_layer = x
		h_layer1 = self.activate(np.dot(x,self.__layer1_weights))
		h_layer2 = self.activate(np.dot(h_layer1,self.__layer2_weights))
		output_layer = h_layer2

		return output_layer


nn =nn()                                                                # Object creation of our model
nn.train(x=train_data,y=train_labels)                # training of our model
print(nn.test(test_data))                                       # testing of our model