# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:41:14 2022

@author: sc2195
"""
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

#function for producing a rolling average data set for a given list - takes data "lis" and a smoothing factor "sf"
def Smoothed(lis, sf):
    smooth = []
    for n in range(len(lis)):
        smooth.append(np.average(lis[max(0, n-sf):n+1]))
    return smooth

#define fundamentals
averages = []
stacksize = 835
'''
for n in range(stacksize):
    
    #load images one by one from tiff stack and convert to np array
    image = io.imread("C:/Users/sc2195/Pictures/CMF_Pics/3.tif", plugin="tifffile", key=n)
    im = np.array(image)
    
    #segment images into four quadrants
    shape = im.shape
    im1 = im[:800, :800]
    im2 = im[:800, 800:]
    im3 = im[800:, :800]
    im4 = im[800:, 800:]
    images = [im1, im2, im3, im4]
    
    #extract average brightness across each image
    av = [np.average(item) for item in images]
    averages.append(av)

#convert gathered data into appropriate format for plotting    
averages = np.array(averages)
data = averages.transpose()
'''

#input skip for development - ensure that only one of the next two rows is commented
#np.savetxt("quadrant_av_intensity_data.csv", data, delimiter=",")
data = np.genfromtxt("quadrant_av_intensity_data.csv", delimiter=",")

x_axis = [n for n in range(len(data[0]))]

#plot in mpl, only necessary for dev purposes
#plt.plot(x_axis, data[0], label = "Image 1")
#plt.plot(x_axis, data[1], label = "Image 2")
#plt.plot(x_axis, data[2], label = "Image 3")
#plt.plot(x_axis, data[3], label = "Image 4")
plt.plot(x_axis, Smoothed(data[2], 4), label = "Image 3, Smoothed")
plt.legend()
plt.show()

   
    

    
    
