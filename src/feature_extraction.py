# -*- coding: utf-8 -*-
import cv2
import numpy as np
import progressbar

def colorhist(im):
    chans = cv2.split(im)
    color_hist = np.zeros((256,len(chans)))
    for i in range(len(chans)):
        color_hist[:,i] = np.histogram(chans[i], bins=np.arange(256+1))[0]/float((chans[i].shape[0]*chans[i].shape[1]))
    return color_hist
    
def get_colorhist_frame(im):
    color_hist = colorhist(im)
    return color_hist
    
def get_colorhist(im_list):
    total = len(im_list)
    bar = progressbar.ProgressBar(maxval=total, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    print 'Generating ColorHist features for [', total, '] images ...'
    bar.start()
    features = {}
    count = 0
    for im_name in im_list:
        im = cv2.imread(im_name)
        color_hist = colorhist(im)
        features[im_name] = color_hist
        bar.update(count)
        count += 1
    bar.finish()
    return features
    
def get_sift_features_frame(im):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, features = sift.detectAndCompute(im, None)
    return features
    
def get_sift_features(im_list):
    """get_sift_features accepts a list of image names and computes the sift descriptos for each image. It returns a dictionary with descriptor as value and image name as key """
    sift = cv2.xfeatures2d.SIFT_create()
    features = {}
    total = len(im_list)
    bar = progressbar.ProgressBar(maxval=total, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    count = 0
    print 'Generating SIFT features for [', total, '] images ...'
    bar.start()
    for im_name in im_list:
        bar.update(count)
        # load grayscale image
        im = cv2.imread(im_name, 0)
        kp, desc = sift.detectAndCompute(im, None)
        features[im_name] = desc
        count += 1
    bar.finish()
    return features
   

