from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# #the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# batch_size = 128
# num_classes = 10
# epochs = 10

# model = Sequential()
# model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4), activation='relu', input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(units=10 ,activation='softmax'))
# model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])

# reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',factor=0.2,patience=5,min_lr=0.001)

# hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
# print("The model has successfully trained")

# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# model.save('model')
# print("Saving the model as model")


model = keras.models.load_model("model")
# visualizer(model, format='png', view=True)
def predict_digit(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class View(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        
        # Creating elements
        self.canvas = tk.Canvas(self, width=280, height=280, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw", font=("Arial Bold", 20))
        self.classify_btn = tk.Button(self, text = "recognise", command = self.classify_handwriting)   
        self.button_clear = tk.Button(self, text = "clear", command = self.clear_all)
       
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text = "Draw")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() 
        rect = win32gui.GetWindowRect(HWND)
        a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=6
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
       
app = View()
mainloop()
