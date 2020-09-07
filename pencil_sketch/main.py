# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:47:03 2020

@author: vamshikrishna Bandari
"""
#loading opencv-python.
import cv2


#reading image in coloured format.
color_img = cv2.imread('2.jpg')

#displaying loading image.
#cv2.imshow('image', color_img)

#waiting the window to close it manually.
#cv2.waitKey(0)

#converting colored image to grey scale image
grey_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)


#displaying loading image.
#cv2.imshow('image', grey_img)

#waiting the window to close it manually.
#cv2.waitKey(0)

#inverting the greyscale image 
invert_img = cv2.bitwise_not(grey_img)

#displaying loading image.
#cv2.imshow('image', invert_img)

#waiting the window to close it manually.
#cv2.waitKey(3000)

#smoothing/blurred of the image 
smoothing_img = cv2.GaussianBlur(invert_img, (21, 21),sigmaX=0, sigmaY=0)


#displaying loading image.
#cv2.imshow('Blured image', smoothing_img)

#waiting the window to close it manually.
#cv2.waitKey(5000)




def func(x, y):
    '''
    A function to divide greyscale image to the blurred or smoothing image
    
    by which to highlight the thick edges.
    
    A technique that oldest photographers used to take images from reels
    '''
    return cv2.divide(x, 255 - y, scale=256)


#calling func to divide and store end result
final_img = func(grey_img, smoothing_img)

#storing the pencil_sketch image
cv2.imwrite('vk_2_pencil_sketch.jpg', final_img)

#displaying loading image.
#cv2.imshow('pencil_sketch', final_img)

#waiting the window to close it manually.
#cv2.waitKey(5000)









