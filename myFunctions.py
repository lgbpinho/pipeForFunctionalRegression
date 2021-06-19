# Define your custom functions and operators in this file
import math

# exp(-x)
def expMinusOne(x):
    return math.exp(-1*x)
    
# avoid division by zero
def safeDiv(x,y):
    return x/(y+0.00001)

# 1/x
def inv(x):
    return 1/(x+0.000001)
