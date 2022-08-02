import imp
# from re import X
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf
# from MLP import MLP
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def SGD_Classifer() :
    n_samples = 400
    noise = 0.02
    factor = 0.5
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    
    sc = StandardScaler()
    model = SGDClassifier(max_iter=300)
    model.fit(x_train, y_train)

    
    y_pred = model.predict(x_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, marker='.')
    plt.title("Train data distribution")
    plt.show()
    print(f"SGD_Accuracy : {sklearn.metrics.accuracy_score(y_test, y_pred)}")
       
    

def CircleClassify(model_type = 'mlp'):
    # generating data
    n_samples = 400
    noise = 0.02
    factor = 0.5
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
    # plt.title("Train data distribution")
    # plt.show()

    ############ Write your codes here - begin
    if model_type == 'mlp':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(3, input_dim=2, activation = 'relu'))
        # model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation = 'sigmoid'))
    elif model_type == 'slp' :
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(2, input_dim=2, activation = 'sigmoid'))
        
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=1,
        epochs=300
    )
    
    results = model.evaluate(x_test, y_test, batch_size=1)
    
    print(f"{model_type}_accuracy:{results[1]}")
    
    ######################Class 분류##############################
    
    y_prob = model.predict(x_test, verbose=0) 
    predicted = y_prob.argmax(axis=-1)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=predicted, marker='.')
    plt.title("Train data distribution")
    plt.show()
    model.summary()

    model.save(".\\circle_model.h5")

    ############ Write your codes here - end


if __name__ == '__main__':
    SGD_Classifer()
    CircleClassify('mlp') #MLP를 사용하려면 "mlp", SLP를 사용하려면 "slp"를 넣어주세요

