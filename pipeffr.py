import yaml
import math
import sys
from random import random
import numpy as np
import myFunctions
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
#import multiprocessing

# A binary tree is used to represent the PPT and programs
class BinaryTree():

    def __init__(self,nodeValue):
      self.left = None
      self.right = None
      self.nodeValue = nodeValue  # PPT: nodeValue is a dict of operators, functions and their 
                                  # probabilities each element is "str": [func,arity.prob]
                                  # ex: {'+': [operator.add,2, 0.32], '*': [operator.mul,2, 0.68]}
      self.constant = 10*random() # to be used at program generation. PPt only

    def getLeftChild(self):
        return self.left
    def getRightChild(self):
        return self.right
    def setNodeValue(self,value):
        self.nodeValue = value
    def getNodeValue(self):
        return self.nodeValue
  

def printTree(tree):
        if tree != None:
            printTree(tree.getLeftChild())
            print(tree.getNodeValue())
            printTree(tree.getRightChild())

# creates a ppt. the leaves have only terminals in it
def createPPT(height,dictNode,dictLeaves):
    if(height>1):
        ppt = BinaryTree(dictNode.copy())
        ppt.left = createPPT(height-1,dictNode,dictLeaves)
        ppt.right = createPPT(height-1,dictNode,dictLeaves)
    elif (height==1):
        ppt = BinaryTree(dictLeaves.copy())
        ppt.left = createPPT(height-1,dictNode,dictLeaves)
        ppt.right = createPPT(height-1,dictNode,dictLeaves)
    else:   
        ppt=None
    return ppt

def printPpt(ppt):
        if ppt != None:
            #printPpt(ppt.getLeftChild())
            print(np.around([v["probability"] for k,v in ppt.getRightChild().getNodeValue().items()],decimals=2))
            #printPpt(ppt.getRightChild())
            
# sample from a ppt node
def sample(dictNode):
    u = random() #super ruim
    probabilities = [dictNode[k]["probability"] for k in dictNode.keys()] # extract probabilities
    elements = [k for k in dictNode.keys()] # extract the elements in the node
    n = len(elements)
    F = np.cumsum(probabilities)
    j=0
    while(u>F[j]and j<(n-1)):
        j=j+1
    return elements[j]

# creates a function from a PPT
def generateFunction(tree, pptNode,height):
    if(height!=0):
        tree.left = BinaryTree(" ") 
        tree.right = BinaryTree(" ")
        chosenElement = sample(pptNode.nodeValue) # choose a symbol
        tree.setNodeValue(chosenElement) # put it in the tree node 

        if(chosenElement=="c"):  # constants will be the one stored in the node or newly generated numbers
            if(pptNode.nodeValue["c"]["probability"]>0.9):
                tree.setNodeValue(pptNode.constant)
            else:
                tree.setNodeValue(1*random())
                
        if(pptNode.nodeValue[chosenElement]["arity"] == 2): # this function needs two arguments
            generateFunction(tree.getLeftChild(),pptNode.getLeftChild(),height-1)
            generateFunction(tree.getRightChild(),pptNode.getRightChild(),height-1)
        elif(pptNode.nodeValue[chosenElement]["arity"] == 1): # this function needs one argument
            generateFunction(tree.getLeftChild(),pptNode.getLeftChild(),height-1),

# evaluate a given function (tree) given the ppt's constant and the variables values
def evaluateFunction(tree,ppt,x):
    value = 0
    if(tree!=None):
        if(not isinstance(tree.nodeValue,str)): # if it is a constant
            return(tree.nodeValue)
        if(ppt.nodeValue[tree.nodeValue]["arity"] == 2):
            # apply the function to the two children
            value = ppt.nodeValue[tree.nodeValue]["function"](evaluateFunction(tree.left,ppt.left,x),evaluateFunction(tree.right,ppt.right,x))
            #print("aridade 2")
        elif(ppt.nodeValue[tree.nodeValue]["arity"] == 1):
            # apply the function to the sole child
            value = ppt.nodeValue[tree.nodeValue]["function"](evaluateFunction(tree.left,ppt.left,x))
            #print("unary")
        else: #it is a variable
            value = x[int(tree.nodeValue[-1])-1] # get the correct variable
    return value

# evaluate the fitness of a variable
#def fit(tree,ppt,x,target):
 #   fitValue = 0
 #   aux = lambda a,t : evaluateFunction(tree,ppt,a)-t # to parallelize
 #   fitValues = Parallel(n_jobs=num_cores)(delayed(aux)(a,t) for a,t in zip(x,target))
#    return sum([z*z for z in fitValues])

# evaluate the fitness of a variable (non parallel version)
def fit(tree,ppt,x,target):
    fitValue = 0
    for a,t in zip(x,target):
        try:
            z = evaluateFunction(tree,ppt,a)-t
            fitValue = fitValue +(z*z)
        except Exception as e:
            #print("Fit valuation failed.")
            fitValue = 10**40
    return fitValue


#adjusts the probabilities in the ppt acording to a rule
def updatePpt_(tree,ppt,learningRate,clr): 
    if(tree.nodeValue!=" " and ppt!=None): # if it is indeed a node
        if(not(isinstance(tree.nodeValue,str))): #if the node is a constant
            aux = ppt.nodeValue["c"]["probability"] #used to normalize probabilities later
            ppt.nodeValue["c"]["probability"] = aux + (1-aux)*clr*learningRate
            ppt.constant = tree.constant
        else:
            aux = ppt.nodeValue[tree.nodeValue]["probability"]
            ppt.nodeValue[tree.nodeValue]["probability"] = aux + (1-aux)*clr*learningRate
        updatePpt_(tree.left,ppt.left,learningRate,clr)
        updatePpt_(tree.right,ppt.right,learningRate,clr)
        for k,v in ppt.nodeValue.items(): #vectors should add up to 1
            v["probability"] = v["probability"]/(1+(1-aux)*clr*learningRate)

# the likelihood of a tree given a ppt
def likelihood(tree,ppt):
    if(tree.nodeValue!=" "): #if that node was used
        ll=1.0
        if(not(isinstance(tree.nodeValue,str))): #if a constant
            ll = (ppt.nodeValue["c"]["probability"])
        elif(ppt.nodeValue[tree.nodeValue]["arity"]==2): #call the function for each argument
            ll = (ppt.nodeValue[tree.nodeValue]["probability"])*likelihood(tree.left,ppt.left)*likelihood(tree.right,ppt.right)
        elif(ppt.nodeValue[tree.nodeValue]["arity"]==1):
            ll = (ppt.nodeValue[tree.nodeValue]["probability"])*likelihood(tree.left,ppt.left)
        elif(ppt.nodeValue[tree.nodeValue]["arity"]==0): #self explanatory, I hope :)
            ll = (ppt.nodeValue[tree.nodeValue]["probability"])
        return ll

# updates the PPT to produce trees similar to a given one
def updatePptTowards(tree,ppt,learningRate,epsilon,bestFitNow,bestFitEver,clr):
    ll = likelihood(tree,ppt)
    targetProbability = ll +(1-ll)*learningRate*(epsilon + bestFitNow)/(epsilon+bestFitEver)
    count=0 
    while(ll<targetProbability and count<5):
        updatePpt_(tree,ppt,learningRate,clr)
        ll = likelihood(tree,ppt)
        count = count+1

# normalize the ppt
def normalizePpt(ppt):
    if(ppt!=None):
        sumOfProbabilities = sum([v["probability"] for k,v in ppt.nodeValue.items()])
        for k,v in ppt.nodeValue.items():
            v["probability"] = v["probability"]/sumOfProbabilities
        normalizePpt(ppt.getRightChild())
        normalizePpt(ppt.getLeftChild())
        

def countNodes(tree):
    if tree == None:
        return  0#-0.5
    if tree!=None:
        return 1 + countNodes(tree.getLeftChild()) + countNodes(tree.getRightChild())
        
# applies the mutation operator
def mutatePpt(bestProgram,ppt,pm,mr):
    if (ppt!=None) and (bestProgram.nodeValue != " "):
        pmp = pm/(len(ppt.nodeValue.keys())*math.sqrt(countNodes(bestProgram)))
        u = random()
        if(u<pmp):
            if(not(isinstance(bestProgram.nodeValue,str))): #if the node is a constant
                ppt.nodeValue["c"]["probability"] = ppt.nodeValue["c"]["probability"] + mr*(1-ppt.nodeValue["c"]["probability"]) 
            else:
                ppt.nodeValue[bestProgram.nodeValue]["probability"] = ppt.nodeValue[bestProgram.nodeValue]["probability"] + mr*(1-ppt.nodeValue[bestProgram.nodeValue]["probability"]) 
        mutatePpt(bestProgram.getRightChild(),ppt.getRightChild(),pm,mr)
        mutatePpt(bestProgram.getLeftChild(),ppt.getLeftChild(),pm,mr)

# print function as a string. needs a sample ppt node for the arity of operators
def printFunction(tree,samplePptNode):
    funcao = ""
    if(tree != None):
        if(not(isinstance(tree.nodeValue,str))): #if the node is a constant
            funcao = "%.2f" % tree.nodeValue
        elif(samplePptNode.nodeValue[tree.nodeValue]["arity"]==2): 
            funcao = "(" + printFunction(tree.getLeftChild(),samplePptNode) +")"+tree.nodeValue+"(" +printFunction(tree.getRightChild(),ppt)+ ")"
        elif(samplePptNode.nodeValue[tree.nodeValue]["arity"]==1): 
            funcao = tree.nodeValue+"(" + printFunction(tree.getLeftChild(),samplePptNode)+")"
        else: # it is a variable
            funcao = tree.nodeValue
    return funcao
    
def copyTree(fromThis):
    toThis = None
    if(fromThis != None):
        toThis = BinaryTree(fromThis.nodeValue)
        toThis.left = copyTree(fromThis.left)
        toThis.right = copyTree(fromThis.right)
    return toThis

if __name__ == "__main__":
#    num_cores = multiprocessing.cpu_count() # number of cores for prallel computation of the fitness values.
                                            # used only by the fit() function
    # read the configurations
    with open("config.yaml", 'r') as config_file:
        try:
            config = yaml.load(config_file)
        except yaml.YAMLError as exc:
            print(exc)

    # read the PIPE's parameters' values
    populationSize = int(sys.argv[3]) #1000 #config.get("populationSize")
    learningRate = config.get("learningRate")
    epsilon = config.get("epsilon")
    eras = config.get("eras")
    updateTowardsEliteProb = config.get("updateTowardsEliteProb")
    clr = config.get("clr")
    pm = config.get("pm")
    mutationRate = config.get("mutationRate")
    nvar = int(sys.argv[2]) #config.get("nvar")
    hmax = config.get("hmax")

    # read the data file
    data = np.loadtxt(sys.argv[1], delimiter=",", usecols=range(nvar+1))
    x =data[:,1:(nvar+1)]
    F =data[:,0]

    # get the operators defined in the configuration file
    dictNodes = config.get("operators")
    dictNodes.update({("x"+str(i)):{"arity": 0, "probability": 1} for i in range(1,nvar+1)})
    dictNodes.update({"c":{"arity": 0, "probability": 1}})
    dictLeaves={("x"+str(i)):{"arity": 0, "probability": 1} for i in range(1,nvar+1)}
    dictLeaves.update({"c": {"arity": 0, "probability": 1}})

    #create the PPT
    ppt = createPPT(hmax,dictNodes,dictLeaves)
    normalizePpt(ppt)

    # getting some variables ready
    bestTreeNow = BinaryTree(" ")
    bestFitNow = float("inf")
    bestTreeEver = BinaryTree(" ")
    bestFitEver =float("inf")
    currentTree = BinaryTree(" ")
    currentFit = 0.0
    print(populationSize)
    for j in range(0,eras): #for each era
        bestFitNow = float("inf") #reset the best fit for this population
        for i in range(0,populationSize): #for each member of the population
            generateFunction(currentTree,ppt,hmax) #generate the function
            currentFit = fit(currentTree,ppt,x,F) #evaluate its fit
            if(currentFit<bestFitNow): #update the best from the population if needed
                bestFitNow = currentFit
                bestTreeNow = copyTree(currentTree)
        #the current population was evaluated by now
        if(bestFitNow<bestFitEver):  #update the best function of all time if needed
                bestFitEver = bestFitNow
                bestTreeEver = copyTree(bestTreeNow)
        #update the PPT
        if(random()<updateTowardsEliteProb):
            updatePptTowards(bestTreeEver,ppt,learningRate,epsilon,bestFitNow,bestFitEver,clr)
            mutatePpt(bestTreeEver, ppt, pm, mutationRate)
            normalizePpt(ppt)
            printTree
        else:
            updatePptTowards(bestTreeNow,ppt,learningRate,epsilon,bestFitNow,bestFitEver,clr)
            mutatePpt(bestTreeNow, ppt, pm, mutationRate)
            normalizePpt(ppt)
        print("Generation {}: Best fit so far: {} for the function {}".format(j, bestFitEver, printFunction(bestTreeEver,ppt)))
    print(printFunction(bestTreeEver,ppt))
    aproximado = [evaluateFunction(bestTreeEver,ppt,a) for a in x]
    plt.scatter(aproximado, F, marker='o')
    plt.ylabel("True value")
    plt.xlabel("Approximate value")
    plt.show()













