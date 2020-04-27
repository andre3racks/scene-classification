import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# plot test and training performance over a given variable
def plot_2D(title, x_label, y_label, test, train, variable, test1=None, train1=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(variable, test, 'r', label="test")
    if train is not None:
        plt.plot(variable, train, 'g', label="train")
    if train1 is not None:
        plt.plot(variable, train1, 'y')
    if test1 is not None:
        plt.plot(variable, test1, 'b')

    plt.legend()
    plt.show()

# 3d plot
def plot_3D(title, x_label, y_label, z_label, xdata, ydata, knn_train, knn_test):
    
    fig = plt.figure()
    ax = Axes3D(fig) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # ax.scatter3D(xdata, ydata, svm_train, c='green', marker='o', label='svm train')
    # ax.scatter3D(xdata, ydata, svm_test, c='red', marker='o', label='svm test')    
    ax.scatter3D(xdata, ydata, knn_train, c='blue', marker='^', label='knn train')  
    ax.scatter3D(xdata, ydata, knn_test, c='red', marker='o', label='knn test')

    ax.legend()   
    plt.show()   

def bar_chart():

    plt.title("Performance Comparison of Dataset Subsets")
    x = np.arange(3)
    prec = [0.502, 0.586, 0.618]
    plt.bar(x, prec)
    plt.ylabel("Average Precision")
    plt.xticks(x, ['all', 'manmade', 'natural'])
    plt.show()

if __name__ == "__main__":

    bar_chart()
    # topics = [i for i in range(3,40,2)] * 13
    # print("topic length = ", len(topics))
    # num_words = [40, 45, 55, 60, 75, 85, 92, 95, 100, 108, 117, 125, 150]
    # NVM = []

    # for num in num_words:
    #     for i in range(19):
    #         NVM.append(num)

    # print("num words len = ", len(NVM))
    # print("range = ", len(topics))

    # fake_data = [0.5 for i in range(0, len(topics))]

    # fake_data = np.asarray(fake_data)
    # plot_3D("Performance of SVM and KNN Classifiers varying K and Z for patch size=11x11", "Number of Topics", "Number of Visual Words", "Accuracy", topics, NVM, fake_data*0.1, fake_data*0.5, fake_data*2, fake_data*3)