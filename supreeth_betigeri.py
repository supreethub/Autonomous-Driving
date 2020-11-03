#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from tqdm import tqdm
import time


# In[1]:


def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]

    # You could import your images one at a time or all at once first, 
    # here's some code to import a single image:
    #frame_num = int(frame_nums[0])
    training_image = np.empty((1500,3840))
    for i in range(1500):
        im_full = cv2.imread(path_to_images + '/' + str(int(frame_nums[i])).zfill(4) + '.jpg')
        im_full = cv2.resize(im_full,(60,64), interpolation = cv2.INTER_AREA)
        im_full = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
        im_full = im_full/255
        im_full = np.ravel(im_full)
        training_image[i] = im_full

    steering_angle_bins = np.zeros((1500, 64))
    bins = np.linspace(-180,180,64)
    pos = np.digitize(steering_angles, bins)
    current = np.zeros((1,64))
    current[0][27:36]=[0.1,0.32,0.61,0.89,1,0.89,0.61,0.32,0.1]
    for i in range(1500):
        steering_angle_bins[i] = np.roll(current,pos[i]-32)
        
    training_angle = steering_angle_bins
    
    #number of iterations
    iterations = 1850
    
    # Train your network here. You'll probably need some weights and gradients!
    X = training_image
    y = training_angle
    NN = NeuralNetwork(Lambda = 0.0001)
    
    
    #Applying Adam Optimizer for Stochaistic Optimization - good default values are:
    #alpha = 0.001
    #beta1 = 0.9
    #beta2 = 0.999
    #epsilon = 10^^-8
    
    a = 1e-3 #alpha
    b1 = 0.3 #beta1
    b2 = 0.95 #beta2
    eps = 1e-8 #epsilon
    
    
    
    grads = NN.computeGradients(X = X, y = y)
    m0 = np.zeros(len(grads)) #initializing first moment vector
    v0 = np.zeros(len(grads)) #initialize second moment vector
    t = 0 #initialize timestep
    losses = []
    mt = m0
    vt = v0
    
    for i in tqdm(range(iterations)):
        t = t + 1
        grads = NN.computeGradients(X = X, y = y)
        mt = b1*mt + (1-b1)*grads #Update biased first moment estimate
        vt = b2*vt + (1-b2)*grads**2 #Update biased second raw moment estimate
        mt_hat = mt/(1-b1**t) #Compute bias-corrected first moment estimate
        vt_hat = vt/(1-b2**t) #Compute bias-corrected second raw moment estimate
        
        params = NN.getParams()
        update_params = params - a*mt_hat/(np.sqrt(vt_hat)+eps) #Update new parameters
        NN.setParams(update_params)
        losses.append(NN.costFunction(X = X, y = y))
    
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    
    ## Perform inference using your Neural Network (NN) here.
    im_full = cv2.imread(image_file)
    im_full = cv2.resize(im_full, (60, 64),interpolation = cv2.INTER_AREA)
    im_full = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
    im_full = im_full/255
    im_full = np.ravel(im_full)
    FeedForward = NN.forward(im_full)
    bins = np.linspace(-180,180,64)
    return bins[np.argmax(FeedForward)]

    
    
class NeuralNetwork(object):
    def __init__(self, Lambda = 0):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        #Regularization
        self.Lambda = Lambda
        
        self.inputLayerSize = 60*64
        self.outputLayerSize = 64
        self.hiddenLayerSize = 32
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


# In[ ]:




