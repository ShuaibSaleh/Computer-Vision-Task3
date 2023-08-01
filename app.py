from flask import Flask, render_template, request
from flask_cors import CORS
import os
import base64  # convert from string to  bits
import json
import cv2
import numpy as np
import time
import calendar
import image as img1
import harrisoperator as Harris
import matplotlib.pyplot as plt
import json
import math
import cv2
import sift as pysift 
import functions as fs


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("harrisoperator.html")




@app.route("/harris", methods=["GET", "POST"])
def harris():
    if request.method == "POST":
        image_data = base64.b64decode( request.form["image_data"].split(',')[1])

        img_path = img1.saveImage(image_data, "harris_img")
        # img_binary = img1.readImg(img_path)

        img = cv2.imread(img_path)
        imggray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # img = cv2.imread(filepath1)
        # imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = time.time()
        r = Harris.getHarrisRespone(imggray)
        corners = Harris.getHarrisIndices(r)
        cornerImg = np.copy(img)
        cornerImg[corners == 1] = [255, 0, 0]
        t2 = time.time()

        # final_img = './static/images/output/output.jpg'
        # plt.savefig(final_img)

        plt.imsave('./static/images/output/output.jpg', cornerImg)
        # return "./static/images/output/output.jpg", t2-t1

        computationTime = t2 - t1

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        output_path = './static/images/output/output.jpg'

        # return json.dumps({1: f'test'})
        return json.dumps({1: f'<img src="{output_path}?t={time_stamp}" id="ApplyEdges" alt="" >', 2: f'<p class="btn btn-success">Computation Time: {round(computationTime, 5)} Seconds</p>'})

    else:
        return render_template("harrisoperator.html")


@app.route("/sift")
def sift():
    return render_template("sift.html")

# siftData





#  sttttttttttttttttttttop here 

@app.route("/siftData", methods=["GET", "POST"])
def siftData():
    path=""

    if request.method == "POST":
        MIN_MATCH_COUNT = 10
        imageA_data = base64.b64decode(request.form["imageA_data"].split(',')[1])
        imageB_data = base64.b64decode(request.form["imageB_data"].split(',')[1])
        path_image1=img1.saveImage(imageA_data ,"image_A")
        path_image2=img1.saveImage(imageB_data ,"image_B")
        image1 = cv2.imread(path_image1, 0)         
        image2 = cv2.imread(path_image2, 0)  
        # Compute SIFT keypoints and descriptors
        kp1, des1 = pysift.computeKeypointsAndDescriptors(image1)
        kp2, des2 = pysift.computeKeypointsAndDescriptors(image2)

        # Initialize and use FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            # Estimate homography between template and scene
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # type: ignore
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) # type: ignore

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

            # Draw detected template in scene image
            h, w = image1.shape
            pts = np.float32([[0, 0],
                            [0, h - 1],
                            [w - 1, h - 1],
                            [w - 1, 0]]).reshape(-1, 1, 2) # type: ignore
            dst = cv2.perspectiveTransform(pts, M)


            image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            h1, w1 = image1.shape
            h2, w2 = image2.shape
            nWidth = w1 + w2
            nHeight = max(h1, h2)
            hdif = int((h2 - h1) / 2)
            newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

            for i in range(3):
                newimg[hdif:hdif + h1, :w1, i] = image1
                newimg[:h2, w1:w1 + w2, i] = image2

            # Draw SIFT keypoint matches
            for m in good:
                pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
                pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
                cv2.line(newimg, pt1, pt2, (255, 0, 0))

            # plt.imshow(newimg)
            # plt.show()
            path = f'./static/images/output/outsift.jpg'
            plt.imsave(path, newimg)

        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        
    
       

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        return json.dumps({1: f'<img src="{path}?t={time_stamp}" id="imageC" alt="" >'})




    else:
        return render_template("sift.html")




@app.route("/SSD_NCC_match", methods=["GET", "POST"])
def SSD_NCC_match():
    path=""

    if request.method == "POST":
        
        imageA_data = base64.b64decode(request.form["imageA_data"].split(',')[1])
        imageB_data = base64.b64decode(request.form["imageB_data"].split(',')[1])
        path_image1=img1.saveImage(imageA_data ,"image_A")
        path_image2=img1.saveImage(imageB_data ,"image_B")
        mthod = int(request.form["method"])
        

        if mthod == 0:
             path,processin_time = fs.match_imgs_using_SSD(path_image1,path_image2)
        elif mthod == 1:
            path,processin_time = fs.match_imgs_using_NCC(path_image1,path_image2)
         
      
       
    
       

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        return json.dumps({1: f'<img src="{path}?t={time_stamp}" id="imageC" alt="" >',2: f'<p class="btn btn-success">Computation Time: {round(processin_time, 5)} Seconds</p>'})




    else:
        return render_template("SSD_NCC_match.html")






if __name__ == "__main__":
    app.run(debug=True , port=5010)
