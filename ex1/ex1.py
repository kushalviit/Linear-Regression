import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab


def computeCost(X,y,theta):
    x_s=np.shape(X)
    m=x_s[0]
    y_s=np.shape(y)
    assert x_s[0]==y_s[0], "Same number of inputs and outputs needed"
    dimension=x_s[1]
    assert dimension==np.shape(theta)[0],"Dimensions of theta and input doesn't match"
    predictions=np.matmul(X,theta)
    J=np.sum(np.power(predictions-y,2))/(2*m)
    return J
          
def gradientDescent(X,y,theta,alpha,num_iterations):
    m=np.shape(X)[0]
    y_num=np.shape(y)[0]
    assert m==y_num," Same number of input and output examples needed"
    dimension=np.shape(X)[1]
    J_history=(np.matrix(np.zeros(num_iterations))).T
    for iter in range(num_iterations):
        delta=np.sum(np.multiply((np.matmul(X,theta)-y),X),axis=0)
        theta=theta-((alpha/m)*delta.T)
        J_history[iter]=computeCost(X,y,theta)
    return [theta,J_history]
        

file_name="ex1data1.txt"
input_signal=np.loadtxt(file_name,delimiter=',')
X=np.matrix(input_signal[:,0])
y=np.matrix(input_signal[:,1])

if np.shape(X)[0]==1:
   X=X.T
if  np.shape(y)[0]==1:
   y=y.T
plt.scatter(X,y)
plt.show()
sh=np.shape(X)
n_ex=sh[0]


X=np.hstack((np.transpose(np.matrix(np.ones(n_ex))),X))
theta=np.transpose(np.matrix(np.zeros(np.shape(X)[1])))

cost=computeCost(X,y,theta)
print("value of J for theta:[0;0] ="+str(cost))

theta=np.transpose(np.matrix([-1,2]))
cost=computeCost(X,y,theta)
print("value of J for theta:[-1;2] ="+str(cost))

iterations = 1500
alpha = 0.01
theta=np.transpose(np.matrix([0,0]))
[theta,J_history] = gradientDescent(X, y, theta, alpha, iterations)

#print("Value of J_history")
#print(str(J_history))
print("Value of theta is :"+str(theta))

predict1=np.matmul(np.matrix(np.array([1,3.5])),theta)
predict1=predict1*10000
print("For population = 35,000, we predict a profit of"+str(predict1))

predict2=np.matmul(np.matrix(np.array([1,7])),theta)
predict2=predict2*10000
print("For population = 70,000, we predict a profit of"+str(predict2))


theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals=np.matrix(np.zeros((len(theta0_vals),len(theta1_vals))))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.matrix(np.array([theta0_vals[i], theta1_vals[j]]));
        t=t.T; 
        J_vals[i,j]=computeCost(X,y,t);

J_vals=J_vals.T;

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm)

ax.zaxis.set_major_locator(LinearLocator(1))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


fig, ax = plt.subplots()
CS = ax.contourf(theta0_vals, theta1_vals, J_vals,locator=ticker.LogLocator())
ax.scatter(theta[0],theta[1])
#plt.clabel(CS, inline=1, fontsize=10)
plt.show()


