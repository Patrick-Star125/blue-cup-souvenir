import pickle
import numpy as np
import cv2
import joblib
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.messagebox
from sklearn.model_selection import train_test_split
from linear_regerssion_mpg import OLSinearRegression
from neural_network_regression_mpg import ANNClassifier
from Decision_tree_mpg import CartRegressionTree
from KNN_mpg import KDTree

# 标准书写模式
"""
Loading data
"""
X = np.loadtxt('auto-mpg.txt', usecols=(1, 2, 3, 4, 5, 6, 7))
y = np.loadtxt('auto-mpg.txt', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# 归一化操作只在神经网络和KNN中需要
X_train2 = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
y_train2 = y_train[:, None]
X_test2 = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
y_test2 = y_test[:, None]

"""
Training
"""
# linear regression
# ols = OLSinearRegression()
# ols.train(X_train, y_train)

# neural network
# ann = ANNClassifier(hidden_layer_sizes=(50,50), eta=0.0001, max_iter=50000, tol=0.00001)
# ann.train(X_train, y_train)

# Decision tree
# crt = CartRegressionTree()
# crt.train(X_train, y_train)

# KDtree
# kd = KDTree(3)
# kd.train(X_train, y_train)

"""
Saving model
"""
# linear regression
# ols.save("mpgmodel_linear_7")

# neural network
# ann.save("mpgmodel_ANN_twolayers")

# Decision tree
# crt.save('mpgmodel_tree_7')

# KDtree
# kd.save('KNN_model/mpgmodel_KDtree_9')

"""
Loading from saved model
"""
# linear regression
ols = OLSinearRegression.from_file('Linear_model/mpgmodel_linear_7')

# neural network
# ann = ANNClassifier.from_file('mpgmodel_ANN_')

# Decision tree
# crt=CartRegressionTree.from_file('mpgmodel_tree')

# KDtree
# kd=KDTree.from_file('mpgmodel_KDtree')

"""
casuall input
"""
# 结果输出函数
def combine_get():
    ip0 = var_cylinder.get()
    ip1 = var_distance.get()
    ip2 = var_hp.get()
    ip3 = var_weight.get()
    ip4 = var_acceleration.get()
    ip5 = var_modelyear.get()
    ip6 = var_place.get()
    ip = []
    ip.append(int(ip0))
    ip.append(int(ip1))
    ip.append(int(ip2))
    ip.append(int(ip3))
    ip.append(int(ip4))
    ip.append(int(ip5))
    ip.append(int(ip6))
    ip = np.array((ip))
    ols = OLSinearRegression.from_file('ANN_model/mpgmodel_ANN_orign_7')
    mpg = ols.predict(ip[None, :])
    mpg_predict = tk.StringVar()
    mpg_predict.set(str(mpg))
    mpg_predict_set = tk.Entry(window, textvariable=mpg_predict, font=('Arial', 14))
    mpg_predict_set.place(x=220, y=350)

# 创建界面
window = tk.Tk()
window.title('mpg predict')
window.geometry('500x500')
canvas = tk.Canvas(window, width=500, height=500, bg='green')
image_file = tk.PhotoImage(file='image/media.io_lrHiatxA.gif')
image = canvas.create_image(250, 0, anchor='n', image=image_file)
canvas.pack(side='top')
tk.Label(window, text='current_model:ANN regression', font=('Arial', 16)).place(x=150,y=400)
tk.Label(window, text='if you want to change the model ,please change it at codefile', font=('Arial', 8)).place(x=150,y=430)
tk.Label(window, text='气缸：', font=('Arial', 14)).place(x=150, y=20)
tk.Label(window, text='行驶里程：', font=('Arial', 14)).place(x=110, y=60)
tk.Label(window, text='马力：', font=('Arial', 14)).place(x=150, y=100)
tk.Label(window, text='重量：', font=('Arial', 14)).place(x=150, y=140)
tk.Label(window, text='加速度：', font=('Arial', 14)).place(x=130, y=180)
tk.Label(window, text='车型年份：', font=('Arial', 14)).place(x=110, y=220)
tk.Label(window, text='产地：', font=('Arial', 14)).place(x=150, y=260)
tk.Label(window, text='mpg：', font=('Arial', 16)).place(x=150, y=350)
var_cylinder = tk.StringVar()
var_distance = tk.StringVar()
var_hp = tk.StringVar()
var_weight = tk.StringVar()
var_acceleration = tk.StringVar()
var_modelyear = tk.StringVar()
var_place = tk.StringVar()
entry_cylinder = tk.Entry(window, textvariable=var_cylinder, font=('Arial', 14))
entry_cylinder.place(x=220, y=20)
entry_distance = tk.Entry(window, textvariable=var_distance, font=('Arial', 14))
entry_distance.place(x=220, y=60)
entry_hp = tk.Entry(window, textvariable=var_hp, font=('Arial', 14))
entry_hp.place(x=220, y=100)
entry_weight = tk.Entry(window, textvariable=var_weight, font=('Arial', 14))
entry_weight.place(x=220, y=140)
entry_acceleration = tk.Entry(window, textvariable=var_acceleration, font=('Arial', 14))
entry_acceleration.place(x=220, y=180)
entry_modelyear = tk.Entry(window, textvariable=var_modelyear, font=('Arial', 14))
entry_modelyear.place(x=220, y=220)
entry_place = tk.Entry(window, textvariable=var_place, font=('Arial', 14))
entry_place.place(x=220, y=260)
btn_predict = tk.Button(window, text='predict', command=combine_get)
btn_predict.place(x=280, y=300)
# 窗口循环
window.mainloop()
"""
Prediction using saved model
"""
# linear regression
y_pred = ols.predict(X_test)

# neural network
# y_pred = ann.predict(X_test)

# Decision tree
# y_pred = crt.predict(X_test)

# KDtree
# y_pred = kd.predict(X_test)
# y_pred=y_pred[:, None]

print(y_pred)
MSE = np.sum((y_test - y_pred) ** 2) / len(y_test)
print(MSE)
MAE = np.sum(abs(y_pred - y_test)) / len(y_test)
print(MAE)

"""
Data Visualization
"""

# def rand(X,y):
#     MAE = []
#     MSE = []
#     for i in range(50):
#         idx = np.arange(len(X))
#         idx=np.random.choice(idx,size=50)
#         # np.random.shuffle(idx)
#         X_test = X[idx]
#         y_test = y[idx]
#         # X_test = X[::4]
#         # y_test = y[::4]
#         y_pred = ols.predict(X_test)
#         a = np.sum(abs(y_pred - y_test)) / len(y_test)
#         s = np.sum((y_test - y_pred) ** 2) / len(y_test)
#         MAE.append(a)
#         MSE.append(s)
#
#     return MAE, MSE


# MAE_all, MSE_all = rand(X_test, y_test)
# print(MAE_all)
# print(MSE_all)

##   free operating   ###
# 此处ols代指任意模型，后模型名没有关系，只是做为模型载入的中间段
# ols5 = OLSinearRegression.from_file('Linear_model/mpgmodel_linear_7')
# def rand5(X,y):
#     MAE = []
#     MSE = []
#     for i in range(50):
#         idx = np.arange(len(X))
#         idx=np.random.choice(idx,size=50)
#         # np.random.shuffle(idx)
#         X_test = X[idx]
#         y_test = y[idx]
#         # X_test = X[::4]
#         # y_test = y[::4]
#         y_pred = ols5.predict(X_test)
#         # y_pred=y_pred[:,None]
#         a = np.sum(abs(y_pred - y_test)) / len(y_test)
#         s = np.sum((y_test - y_pred) ** 2) / len(y_test)
#         MAE.append(a)
#         MSE.append(s)
#
#     return MAE, MSE
# ols6 = OLSinearRegression.from_file('Decision_tree_model/mpgmodel_tree_7')
# def rand6(X,y):
#     MAE = []
#     MSE = []
#     for i in range(50):
#         idx = np.arange(len(X))
#         idx=np.random.choice(idx,size=50)
#         # np.random.shuffle(idx)
#         X_test = X[idx]
#         y_test = y[idx]
#         # X_test = X[::4]
#         # y_test = y[::4]
#         y_pred = ols6.predict(X_test)
#         # y_pred = y_pred[:, None]
#         a = np.sum(abs(y_pred - y_test)) / len(y_test)
#         s = np.sum((y_test - y_pred) ** 2) / len(y_test)
#         MAE.append(a)
#         MSE.append(s)
#
#     return MAE, MSE
# ols7 = OLSinearRegression.from_file('KNN_model/mpgmodel_KDtree_7')
# def rand7(X,y):
#     MAE = []
#     MSE = []
#     for i in range(50):
#         idx = np.arange(len(X))
#         idx=np.random.choice(idx,size=50)
#         # np.random.shuffle(idx)
#         X_test = X[idx]
#         y_test = y[idx]
#         # X_test = X[::4]
#         # y_test = y[::4]
#         y_pred = ols7.predict(X_test)
#         # y_pred = y_pred[:, None]
#         a = np.sum(abs(y_pred - y_test)) / len(y_test)
#         s = np.sum((y_test - y_pred) ** 2) / len(y_test)
#         MAE.append(a)
#         MSE.append(s)
#
#     return MAE, MSE
# ols8 = OLSinearRegression.from_file('ANN_model/mpgmodel_ANN_orign_7')
# def rand8(X,y):
#     MAE = []
#     MSE = []
#     for i in range(50):
#         idx = np.arange(len(X))
#         idx=np.random.choice(idx,size=50)
#         # np.random.shuffle(idx)
#         X_test = X[idx]
#         y_test = y[idx]
#         # X_test = X[::4]
#         # y_test = y[::4]
#         y_pred = ols8.predict(X_test)
#         # y_pred = y_pred[:, None]
#         a = np.sum(abs(y_pred - y_test)) / len(y_test)
#         s = np.sum((y_test - y_pred) ** 2) / len(y_test)
#         MAE.append(a)
#         MSE.append(s)
#
#     return MAE, MSE
# ols9 = OLSinearRegression.from_file('ANN_model/mpgmodel_ANN_orign_9')
# def rand9(X,y):
#     MAE = []
#     MSE = []
#     for i in range(50):
#         idx = np.arange(len(X))
#         idx=np.random.choice(idx,size=50)
#         # np.random.shuffle(idx)
#         X_test = X[idx]
#         y_test = y[idx]
#         # X_test = X[::4]
#         # y_test = y[::4]
#         y_pred = ols9.predict(X_test)
#         # y_pred = y_pred[:, None]
#         a = np.sum(abs(y_pred - y_test)) / len(y_test)
#         s = np.sum((y_test - y_pred) ** 2) / len(y_test)
#         MAE.append(a)
#         MSE.append(s)
#
#     return MAE, MSE
#
#
# ###     draw some charts     ###
# plt.figure()
# MAE_all5,MSE_all5=rand5(X_test,y_test)
# MAE_all6,MSE_all6=rand6(X_test,y_test)
# MAE_all7,MSE_all7=rand7(X_test,y_test)
# MAE_all8,MSE_all8=rand8(X_test,y_test)
# MAE_all9,MSE_all9=rand9(X_test,y_test)
# plt.plot(MAE_all5,'b-',label='MAE_all_5')
# plt.plot(MAE_all6,'g-',label='MAE_all_6')
# plt.plot(MAE_all7,'r-',label='MAE_all_7')
# plt.plot(MAE_all8,'y-',label='MAE_all_8')
# plt.plot(MAE_all9,'k-',label='MAE_all_9')
# plt.legend(loc='best')
# plt.title('the training ratio of all',fontsize=16)
# plt.ylabel('MAE')
# plt.text(20.0,3.5,'mean_Linear:%f'%np.mean(MAE_all5),color='b')
# plt.text(20.0,3.2,'mean_Decisiontree:%f'%np.mean(MAE_all6),color='g')
# plt.text(20.0,2.9,'mean_KDtree:%f'%np.mean(MAE_all7),color='r')
# plt.text(20.0,2.7,'mean_ANN:%f'%np.mean(MAE_all8),color='y')
# plt.text(20.0,2.6,'mean_9:%f'%np.mean(MAE_all9),color='k')
# plt.show()
