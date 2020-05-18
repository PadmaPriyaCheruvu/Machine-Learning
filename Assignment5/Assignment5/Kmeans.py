import sys
import numpy as np
from skimage import io, img_as_float
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

def kMeans(imgVect, k, num_iterations):
    labels = np.full((imgVect.shape[0],), -1)

    prototypes = np.random.rand(k, 3)
    for i in range(num_iterations):
        points = [None for k_i in range(k)]
        for rgb_i, rgb in enumerate(imgVect):
            rgb_row = np.repeat(rgb, k).reshape(3, k).T
            closest_label = np.argmin(np.linalg.norm(rgb_row - prototypes, axis=1))
            labels[rgb_i] = closest_label
            if (points[closest_label] is None):
                points[closest_label] = []
            points[closest_label].append(rgb)

        for k_i in range(k):
            if (points[k_i] is not None):
                new_prototype = np.asarray(points[k_i]).sum(axis=0) / len(points[k_i])
                prototypes[k_i] = new_prototype

    return (labels, prototypes)

def calc_centroids(X,index,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        c_i = index
        c_i = c_i.astype(int)
        total_number = sum(c_i);
        c_i.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(c_i,1,n)
        c_i = np.transpose(c_i)
        total = np.multiply(X,total_matrix)
        centroids[i] = (1/total_number)*np.sum(total,axis=0)
    return centroids

def closest_centroids(X,c):
    K = np.size(c,0)
    index = np.zeros((np.size(X,0),1))
    array = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        temp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,temp),2)
        a = np.sum(b,axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0),1))
        array = np.append(array, a, axis=1)
    array = np.delete(array,0,axis=1)
    index = np.argmin(array, axis=1)
    return index

def plot_by_color(name, imgVect):
    fig = plt.figure()
    axes = Axes3D(fig)
    for rgb in imgVect:
        axes.scatter(rgb[0], rgb[1], rgb[2], c=rgb, marker='o')
    axes.set_xlabel('Red')
    axes.set_ylabel('Green')
    axes.set_zlabel('Blue')
    fig.savefig(name + '.png')


def plot_by_label(name, imgVect, labels, prototypes):
    fig = plt.figure()
    axes = Axes3D(fig)
    for rgb_i, rgb in enumerate(imgVect):
        axes.scatter(rgb[0], rgb[1], rgb[2], c=prototypes[labels[rgb_i]], marker='o')
    axes.set_xlabel('Red')
    axes.set_ylabel('Green')
    axes.set_zlabel('Blue')
    fig.savefig(name + '.png')


if __name__ == '__main__':
    
    #Get command line args
    picture =sys.argv[1]
    K=int(sys.argv[2])
    iterations=int(sys.argv[3])
    
    info = os.stat(J)
    print("Image size before compression: ",info.st_size/1024,"KB")
    
    image = io.imread(picture)[:, :, :3] 
    image = img_as_float(image)
    imgDimensions = image.shape
    imgName = image
    imgVect = image.reshape(-1, image.shape[-1])
    print('Starting iterations')
    labels, color_centroids = kMeans(imgVect, k=K, num_iterations=iterations)
    output = np.zeros(imgVect.shape)
    for i in range(output.shape[0]):
        output[i] = color_centroids[labels[i]]
    output = output.reshape(imgDimensions)
    
    io.imsave('Compressed.jpg' , output)
    print('Image saved')
    
    info = os.stat('Compressed.jpg')
    print("Image size after compression: ",info.st_size/1024,"KB")