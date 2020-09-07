# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:00:53 2020

@author: vamshikrishna Bandari

"""
import cv2


video = cv2.VideoCapture(0)



while(True):
    ret, frame = video.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()

cv2.destroyAllWindows()