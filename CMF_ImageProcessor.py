# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:41:14 2022

@author: sc2195
"""
import sys
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.signal as signal

#function for producing a rolling average data set for a given list - takes data "lis" and a smoothing factor "sf"
def Smoothed(lis, sf):
    smooth = []
    for n in range(len(lis)):
        smooth.append(np.average(lis[max(0, n-sf):n+1]))
    return smooth

#function to take a data set and find local maxima. Requires tuning of range "r", and prominance "p" to decide 
#upon significance of peaks
def PickMax(lis, r, p):
    maxi = []
    for n in range(len(lis)):
        
        #find start and end points which won't throw errors or accidentally wrap
        start = max([0, n-r])
        end = min([len(lis), n+r])
        
        #calculate the prominance of a peak in the series (how much it stands above the local area)
        prominance = lis[n] - np.average(lis[start:end])
        
        #find points which are both maxima and sufficiently prominant
        if max(lis[start:end]) == lis[n] and prominance>p:
            features = [n, lis[n]]
            maxi.append(features)
    
    maxi = np.asarray(maxi)
    
    if maxi.ndim == 1:
        sys.exit("ERROR - PickMax found no maxima in selected range.")
    else:
        return maxi

#function for returning start and end points on plugs. Tuned for selection on channel 4 (bottom right quadrant).
def PlugPick(series):
    #pick maxima, then process for plug regions
    maxima = PickMax(series, 5, 1000)
    plug_s = []
    plug_e = []
    #find pairs of maxima, then save as a tuple, [picture index, intensity]
    for n in range(1,len(maxima)):
        if maxima[n-1][1] < 11000:
            
            #tune the integers at the end of the next two expressions to get the right start/end points
            s_index = maxima[n-1][0]-2
            e_index = maxima[n][0]-5
            
            plug_s.append([s_index,series[int(s_index)]])
            plug_e.append([e_index,series[int(e_index)]])
            
    return plug_s, plug_e, maxima

'''
#define fundamentals
averages = []
stacksize = 835

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

#pick plugs
plug_s, plug_e, maxima = PlugPick(Smoothed(data[3], 4))

#commented below are plotting scripts - useful for bug fixes in the plug id stage.
'''
#processing dataframes for plotting ONLY
maxima = maxima.transpose()
plug_s = np.asarray(plug_s).transpose()
plug_e = np.asarray(plug_e).transpose()

x_axis = [n for n in range(len(data[0]))]

#plot in mpl, only necessary for dev purposes
#plt.plot(x_axis, data[0], label = "Image 1")
#plt.plot(x_axis, data[1], label = "Image 2")
#plt.plot(x_axis, data[2], label = "Image 3")
plt.plot(x_axis, data[3], label = "Image 4")
#plt.plot(x_axis, Smoothed(data[2], 4), label = "Image 3, Smoothed")
#plt.plot(x_axis, Smoothed(data[3], 4), label = "Image 4, Smoothed") 
#plt.scatter(maxima[0], maxima[1], c="Red", label="Image 4 Max")
plt.scatter(plug_s[0], plug_s[1], c="Green", label="Plug Start")
plt.scatter(plug_e[0], plug_e[1], c="Purple", label="Plug End")
plt.legend()
plt.show()
'''

starts = [int(n) for n in np.asarray(plug_s).transpose()[0]]
ends = [int(n) for n in np.asarray(plug_e).transpose()[0]]

#iterate through plugs, load all images in each plug and begin processing
data = []
for n in range(len(starts)):
    plug_data = []
    for m in range(starts[n], ends[n]+1):
        
        #load images one by one from tiff stack and convert to correct format for cv2
        im = io.imread("C:/Users/sc2195/Pictures/CMF_Pics/3.tif", plugin="tifffile", key=m)
        im = np.array(im)
        
        #break into quadrants, make each quadrant its own image
        ims = [im[:800, :800], im[:800, 800:], im[800:, :800], im[800:, 800:]]
        ims = [np.array(255*(im/np.max(im)), dtype = np.uint8) for im in ims[1:]]
        
        quads = []
        for src_img in ims:
            color_img = cv.cvtColor(src_img,cv.COLOR_GRAY2BGR)
            circles_img = cv.HoughCircles(src_img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=30,maxRadius=50)
            
            #skip clause in the case that no circles are found
            if circles_img is None:
                continue
            
            circles_img = np.uint16(np.around(circles_img))[0]
            
            bubbles = []
            for circle in circles_img:
                x, y, r = circle.astype(np.int32)
                roi = src_img[y - r: y + r, x - r: x + r]
                
                # generate and apply mask, calculate average brightness
                width, height = roi.shape[:2]
                mask = np.zeros((width, height), roi.dtype)
                cv.circle(mask, (int(width / 2), int(height / 2)), r, (255, 255, 255), -1)
                dst = cv.bitwise_and(roi, mask)
                
                #weird break clause required - some circles appear to be entirely dark?
                if dst is None:
                    continue
                
                av_brightness = np.mean(dst[np.where(dst!=0)])
                
                #implement noise variance estimation - DOI: 10.1006/cviu.1996.0060, generate noise metric
                M = [[1, -2, 1],[-2, 4, -2],[1, -2, 1]]
                sigma = np.sum(np.sum(np.absolute(signal.convolve2d(dst, M))))
                sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (width-2) * (height-2))
                
                bubbles.append([av_brightness, sigma])           
            
            #image printing for bug fixes in the image analysis stage
            if np.mean(np.asarray(bubbles).transpose()[1]) > 8:
                print(circles_img)
                for i in circles_img[:]:
                    cv.circle(color_img,(i[0],i[1]),i[2],(0,255,0),2)
                    cv.circle(color_img,(i[0],i[1]),2,(0,0,255),3)
                
                #cv.imshow('Original Image',src_img)
                cv.imshow('Detected Circles',color_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
            
            quads.append(bubbles)
        plug_data.append(quads)
    data.append(plug_data)

#flatten data. Cols = (1)plug, (2)image, (3)quadrant, (4)bubble, (5)brightness, (6)noise
flat_data = []
for o in range(len(data)):
    for a in range(len(data[o])):
        for b in range(len(data[o][a])):
            for c in range(len(data[o][a][b])):
                flat_data.append([o, a, b, c, data[o][a][b][c][0], data[o][a][b][c][1]])
     
#print test plug data to csv
#flat_data = np.asarray(flat_data)
#np.savetxt("test_postprocessing_data.csv", flat_data, delimiter=",")