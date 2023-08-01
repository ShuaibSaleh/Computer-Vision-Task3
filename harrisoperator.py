import cv2
import numpy as np
import matplotlib.pyplot as plt


def getHarrisRespone(imggray):
    # Calculation of Sobelx
    sobelx = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=5)
    # Calculation of Sobely
    sobely = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=5)
    # Apply GaussianBlue for noise cancellation
    Ixx = cv2.GaussianBlur(src=sobelx ** 2, ksize=(5, 5), sigmaX=0)
    Ixy = cv2.GaussianBlur(src=sobely * sobelx, ksize=(5, 5), sigmaX=0)
    Iyy = cv2.GaussianBlur(src=sobely ** 2, ksize=(5, 5), sigmaX=0)

    k = 0.05
    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
    # get r
    harris_response = detA - k * traceA ** 2
    return harris_response


def getHarrisIndices(harrisRes):
    # find edges and corners using r
    #Edge : r < 0
    #Corner : r > 0
    #Flat: r = 0
    threshold = 0.01
    harrisRecsopy = np.copy(harrisRes)
    rMatrix = cv2.dilate(harrisRecsopy, None)
    rMax = np.max(rMatrix)
    corner = np.array(harrisRes > (rMax*threshold), dtype="int8")
    return corner
