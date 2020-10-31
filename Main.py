import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from Logistc_model import model

#显示一张图片
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes=load_dataset()
# index=25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y= "+str(train_set_y[:,index])+", it's a "+ classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"picture.")

#查看整个数据集的信息
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

#数据集规范化,X按行展开为m*n,Y也为列向量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)
train_set_y=train_set_y.T
test_set_y=test_set_y.T
# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#数据归一化
train_set_x=train_set_x_flatten/256
test_set_x=test_set_x_flatten/256

#开始训练
# d=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.01,lamda=1,print_cost=True)

# #画出学习曲线
# costs=np.squeeze(d["costs"])
# plt.plot(costs)
# plt.xlabel('iterations(per hundreds')
# plt.ylabel('cost')
# plt.title("Learning rate ="+str(d["Learning_rate"]))
# plt.show()

''' 测试不同学习率下的学习结果 '''
learning_rates=[0.001,0.01,0.1]
models = {} #创建字典
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i,lamda=1.0, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["Learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
