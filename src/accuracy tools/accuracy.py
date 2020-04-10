#!/usr/bin/env python

import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import feature_extraction as ft
import sys
import image_search
import os, os.path
from PIL import Image
from prettytable import PrettyTable

# global variables
# vocabularies
colorhist_features = None
geo_features = None
# results
geo_candidates = None
colorhist_candidates = None

# Command line parsing is handled by the ArgumentParser object

features = ['colorhist']

parser = argparse.ArgumentParser(description="Query tool to query the database created by the database tool (dbt.py). Retrieve images based on image content and metadata.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("database", help="Path to the database to execute the query on.")
parser.add_argument("query", help="Query Video")

#   Optional arguments
parser.add_argument("--candidates", help="The number of candidates to consider for each query", default = "1")
default_images_folder = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Images/images'))
parser.add_argument("--images", help="The folder used to create the database images. Used for displaying results.", default=default_images_folder)

args = parser.parse_args()

# Get file name without extension
base = os.path.splitext(args.database)[0]


# Helper functions
# ======================================= 

def print_results(results):
    tab = PrettyTable()
    tab.field_names = ["nk", "oj", "rh", "overall"]
    
    accuracy_nk = results["nk"]["correct"]/(results["nk"]["total"] * 1.0)
    accuracy_oj = results["oj"]["correct"]/(results["oj"]["total"] * 1.0)
    accuracy_rh = results["rh"]["correct"]/(results["rh"]["total"] * 1.0)
    total = results["nk"]["total"] + results["oj"]["total"] + results["rh"]["total"]
    accuracy_overall = results["overall"] / (total * 1.0)
    
    tab.add_row([accuracy_nk, accuracy_oj, accuracy_rh, accuracy_overall])
    print(tab)


def get_prediction(distances, filenames):
    temp = {
        "nk" : 0,
        "oj" : 0,
        "rh" : 0
    }
    for i in range(len(distances)):
        temp[filenames[i][6:8]] += 1/distances[i]
    
    res = max(temp, key = temp.get)
    return res

# Starting point of the script
# =======================================

if __name__ == '__main__':
   
    print '\nMulti Media Analysis Query Tool'
    print '================================\n'
    print "Query the database with [", args.query, "]"
    db_name = args.database
    search = image_search.Searcher(db_name)
    locations_counts = {
        "nk" : 0,
        "rh" : 0,
        "oj" : 0
    }
    
    fname = base + '_colorhist.pkl'
    # Load all colorhistogram features of our training data
    with open(fname, 'rb') as f:
        colorhist_features = pickle.load(f)
    
    imgs = []
    preds = []
    path = "../Images/data/testing/"
    test_data = []

    # Gather test data paths
    for f in os.listdir(path):
        imgs.append(os.path.join(path, f))
      
    # Gather color histograms of test data
    color_hists = ft.get_colorhist(imgs)
    
    # Gather test data predictions and color hists
    for i in range(len(imgs)):
        test_data.append({
            "location" : imgs[i][29:31],
            "color_hist" : color_hists[imgs[i]]
        })
    
    # Set up a flexible structure to gather results
    results = {
    
        "nk" : {
            "correct" : 0,
            "total" : 0
        },
        
        "oj" : {
            "correct" : 0,
            "total" : 0
        },

        "rh" : {
            "correct" : 0,
            "total" : 0
        },
        
        "overall" : 0
    }
    
    total_count = len(imgs)
    
    for query in test_data:

        expected_location = query["location"]
        colorhist_query = query["color_hist"]
        
        # Compare the query colorhist with the dataset and retrieve an ordered list of candidates
        colorhist_candidates = search.candidates_from_colorhist(colorhist_query, colorhist_features)  
       
        # If candidates exists, show the top N candidates
        N = int(args.candidates)

        distances = colorhist_candidates[1][0:N]
        filenames = colorhist_candidates[0][0:N]
        
        predicted_location = get_prediction(distances, filenames)
              
        results[expected_location]["total"] += 1
        
        if (predicted_location == expected_location):
            results["overall"] += 1
            results[predicted_location]["correct"] += 1

    print_results(results)
    

