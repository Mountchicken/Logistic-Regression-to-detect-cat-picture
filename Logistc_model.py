import numpy as np
"""
对于我的这个模型，请参照笔记中的吴恩达机器学习matlab一对多篇，对矩阵的要求如下
1.X为将每个样本按行展开后的数据，为m*n
2.Y为一个列向量，为m*1
3.w为一个列向量，为n*1
4.b为单个元素
"""
#定义sigmod函数
def sigmod(z):
    return 1.0/(1+np.exp(-z))

#初始化权重w(列向量)以及偏置单元b
def initialize_with_zero(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b

#前向传播与反向传播
def propagate(w,b,X,Y,lamda):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    lamda--paramaters for regularization

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot() 
    """
    
    #Forward Propagation
    m=X.shape[0] 
    z=np.dot(X,w)+b 
    A=sigmod(z) #类似于h0x
    cost=-np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))/m+lamda*sum(w**2)/m

    #Backward Propagation
    dw=np.dot(X.T,A-Y)/m+lamda*w/m #带正则化项
    db=np.sum(A-Y)/m
    grads={"dw":dw,"db":db}
    return grads,cost

# w, b, X, Y,lamda= np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]).T, np.array([[1],[0]]),0
# grads,cost = propagate(w, b, X, Y,lamda)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))

#梯度下降函数
def Optimization(w,b,X,Y,num_iterations,learning_rate,lamda,print_cost):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    lamda -- regularization parameters
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y,lamda)
        # Receive derivatives from propagate
        dw=grads["dw"]
        db=grads["db"]
        #upgrade the weigths and bias
        w=w-learning_rate*dw
        b=b-learning_rate*db

        #record the costs
        if i%100==0:
            costs.append(cost)

        #Print the cost every 100 trainging examples
        if print_cost and i%100==0:
            print("Cost afetr iteration %i:%.3f"%(i,cost))
    parms={"w":w,"b":b}
    grads={"dw":dw,"db":db}
    return parms,grads,costs

def predict(w,b,X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (number of examples,num_px * num_px * 3)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    
    '''
    m=X.shape[0]
    Y_prediction=np.zeros((m,1))
    A=sigmod(np.dot(X,w)+b)
    Y_prediction=np.round(A) #大于0.5判为1，小于0.5判为0
    return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.1,lamda=0.0,print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (m_train,num_px * num_px * 3)
    Y_train -- training labels represented by a numpy array (vector) of shape (m_train,1)
    X_test -- test set represented by a numpy array of shape (m_test,num_px * num_px * 3)
    Y_test -- test labels represented by a numpy array (vector) of shape (m_test,1)
    lamda --regularization parameters
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    w,b=initialize_with_zero(X_train.shape[1])
    parameters,grads,costs=Optimization(w,b,X_train,Y_train,num_iterations,learning_rate,lamda,print_cost)
    w=parameters["w"]
    b=parameters["b"]
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)
    #打印训练后的准确性
    print("训练集准确性: ",format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100),"%")
    print("测试集准确性: ",format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100),"%")
    d={
        "costs":costs,
        "Y_prediction_train_rate":1-np.mean(np.abs(Y_prediction_train-Y_train)),
        "Y_preditcion_test_rate":1-np.mean(np.abs(Y_prediction_test-Y_test)),
        "w":w,
        "b":b,
        "Learning_rate":learning_rate
      }
    return d