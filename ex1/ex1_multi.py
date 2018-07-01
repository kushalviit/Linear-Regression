import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab

def normalEqn(X, y):
    theta =np.matrix( np.zeros(np.shape(X)[1])).T
    theta=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T,X)),X.T),y)
    return [theta]

def featureNormalize(X):
    X_norm=np.zeros(np.shape(X))
    mu=np.zeros(np.shape(X)[1])
    std=np.zeros(np.shape(X)[1])
    mu=X.mean(0)
    std=X.std(0)
    X_norm=(X-mu)/std
    return [X_norm,mu,std]

def computeCostMulti(X,y,theta):
    x_s=np.shape(X)
    m=x_s[0]
    y_s=np.shape(y)
    assert x_s[0]==y_s[0], "Same number of inputs and outputs needed"
    dimension=x_s[1]
    assert dimension==np.shape(theta)[0],"Dimensions of theta and input doesn't match"
    predictions=np.matmul(X,theta)
    J=np.sum(np.power(predictions-y,2))/(2*m)
    return J
          
def gradientDescentMulti(X,y,theta,alpha,num_iterations):
    m=np.shape(X)[0]
    y_num=np.shape(y)[0]
    assert m==y_num," Same number of input and output examples needed"
    dimension=np.shape(X)[1]
    J_history=(np.matrix(np.zeros(num_iterations))).T
    for iter in range(num_iterations):
        delta=np.sum(np.multiply((np.matmul(X,theta)-y),X),axis=0)
        theta=theta-((alpha/m)*delta.T)
        J_history[iter]=computeCostMulti(X,y,theta)
    return [theta,J_history]



file_name="ex1data2.txt"
input_signal=np.loadtxt(file_name,delimiter=',');
X=np.matrix(input_signal[:,0:2])
y=np.matrix(input_signal[:,2]).T
m = np.shape(y)[0]

[X,mu,std]=featureNormalize(X)
X=np.hstack((np.transpose(np.matrix(np.ones(m))),X))
theta=np.transpose(np.matrix(np.zeros(np.shape(X)[1])))

alpha=0.01
num_iters = 1500
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
print("Theta Computed from gradientDescentMulti:")
print(theta)

iter_number=np.linspace(1,len(J_history),num=len(J_history))
plt.plot(iter_number,J_history)
plt.show()


T_X=np.matrix(np.array([1650,3]))
T_X=(T_X-mu)/std;
T_X=np.hstack((np.transpose(np.matrix(np.ones(1))),T_X))
price = np.matmul(T_X,theta)
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):")
print(price)

del X,y,theta,price,T_X
file_name="ex1data2.txt"
input_signal=np.loadtxt(file_name,delimiter=',');
X=np.matrix(input_signal[:,0:2])
y=np.matrix(input_signal[:,2]).T
X=np.hstack((np.transpose(np.matrix(np.ones(m))),X))
[theta]=normalEqn(X,y)

print("Theta Computed from Normal Eqn:")
print(theta)

T_X=np.matrix(np.array([1,1650,3]))
price = np.matmul(T_X,theta)
print("Predicted price of a 1650 sq-ft, 3 br house (using Normal Eqn):")
print(price)
