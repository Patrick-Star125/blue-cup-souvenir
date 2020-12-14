import pickle
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regerssion_mpg import OLSinearRegression
import tkinter as tk
from PIL import Image
# import tensorflow as tf


# class Car:  # 外部类
#     class Door:  # 内部类
#         def open(self):
#             print('open door')
#
#     class Wheel:
#         def run(self):
#             print('car run')
#
#
# if __name__ == "__main__":
#     car = Car()  # 实例化外部类
#     backDoor = Car.Door()  # 实例化内部类 第一种方法
#
#     frontDoor = car.Door()  # 因为car已经实例化外部类，再次实例化Car的内部类 第二种方法
#     backDoor.open()
#     frontDoor.open()
#     wheel = car.Wheel()  # car已经实例化外部类，Wheel()再次实例化内部类
#     wheel.run()  # 调用内部类的方法


mnist=tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255,x_test/255

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test,y_test,verbose=2)