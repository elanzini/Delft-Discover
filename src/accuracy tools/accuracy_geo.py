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
import metadata_distance 
import pandas as pd
import seaborn as sn

# global variables
DEBUG = False
# vocabularies
colorhist_features = None
geo_features = None
# results
geo_candidates = None
colorhist_candidates = None

# Command line parsing is handled by the ArgumentParser object

features = ['colorhist']

parser = argparse.ArgumentParser(description="Query tool to query the database created by the database tool (dbt.py). Retrieve images based on image content and metadata.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("database_colorhist", help="Path to the database of color histograms to execute the query on.")
parser.add_argument("database_geo", help="Path to the database of geo metadata to execute the query on.")
parser.add_argument("query", help="Query Video")
#   Optional arguments
parser.add_argument("--candidates", help="The number of candidates to consider for each query", default = "1")
default_images_folder = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Images/images'))
parser.add_argument("--images", help="The folder used to create the database images. Used for displaying results.", default=default_images_folder)

args = parser.parse_args()

# Get file name without extension
base = os.path.splitext(args.database_colorhist)[0]
base_geo = os.path.splitext(args.database_geo)[0]


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
    
def get_prediction_with_geo(distances_colorhist, filenames_colorhist, distances_geo, filenames_geo):
    
    temp = {
        "nk" : 0,
        "oj" : 0,
        "rh" : 0
    }
    
    # Dealing with division by zero
    epsilon = 0.0001
    weight_colorhist = 0.5
    weight_geo = 0.5
    
    for i in range(len(distances_colorhist)):
        if DEBUG:
            print("Color hist distance: " + str(distances_colorhist[i]) + " and predicted " + filenames_colorhist[i][6:8])
            print("Geo distance: " + str(distances_geo[i]) + " and predicted " + filenames_geo[i][30:32])
        temp[filenames_colorhist[i][6:8]] += weight_geo/(distances_colorhist[i] + epsilon)
        temp[filenames_geo[i][30:32]] += weight_geo/(distances_geo[i] + epsilon)
    
    res = max(temp, key = temp.get)
    return res
    
def get_candidates_geo(geo_query, get_features):
    
    # Query has geo data
    if metadata_distance.has_geotag(geo_query):
        distances = []
        for key in geo_features.keys():
            candidate_metadata = geo_features[key]
            if metadata_distance.has_geotag(candidate_metadata):
                distances.append((key, metadata_distance.compute_geographic_distance(geo_query, candidate_metadata)))
            
        geo_candidates = sorted(distances, key = lambda x: x[1])
    # Query has no geo data
    else:
        geo_candidates = None
    
    return geo_candidates 
         
def get_confusion_matrix(data):
    df = pd.DataFrame(data, columns=['actual','pred'])
    confusion_matrix = pd.crosstab(df['actual'], df['pred'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True, cmap = 'YlGnBu')
    plt.show()
    

# Starting point of the script
# =======================================

if __name__ == '__main__':
   
    print '\nMulti Media Analysis Query Tool'
    print '================================\n'
    print "Query the database with [", args.query, "]"
    db_name = args.database_colorhist
    search = image_search.Searcher(db_name)
    locations_counts = {
        "nk" : 0,
        "rh" : 0,
        "oj" : 0
    }
    
    # Load geodata
    fname_geo = base_geo + '_meta.pkl'
    with open(fname_geo, 'rb') as f:
        geo_features = pickle.load(f)

    # Load colorhist
    fname = base + '_colorhist.pkl'
    # Load all colorhistogram features of our training data
    with open(fname, 'rb') as f:
        colorhist_features = pickle.load(f)
    
    # Top N candidates
    N = int(args.candidates)
    preds = {
        "actual" : [],
        "pred" : []
    }
    
    imgs = []
    path = "../Images/data/testing/"
    test_data = []

    # Gather test data paths
    for f in os.listdir(path):
        imgs.append(os.path.join(path, f))
      
    # Gather color histograms of test data
    color_hists = ft.get_colorhist(imgs)
    
    # Gather geo data
    metadata = ft.extract_metadata(imgs)
    
    # Gather test data predictions and color hists
    for i in range(len(imgs)):
        test_data.append({
            "location" : imgs[i][29:31],
            "color_hist" : color_hists[imgs[i]],
            "geo" : metadata[imgs[i]]
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
        geo_query = query["geo"]
        
        # Compare the query colorhist with the dataset and retrieve an ordered list of candidates
        colorhist_candidates = search.candidates_from_colorhist(colorhist_query, colorhist_features)

        distances = colorhist_candidates[1][0:N]
        filenames = colorhist_candidates[0][0:N]
        
        # Compute candidates for geo
        geo_candidates = get_candidates_geo(geo_query, geo_features)
       
        if geo_candidates is not None:
            filenames_geo = [x[0] for x in geo_candidates][0:N]
            distances_geo = [x[1] for x in geo_candidates][0:N]
            predicted_location = get_prediction_with_geo(distances, filenames, distances_geo, filenames_geo)
            
        else:
            predicted_location = get_prediction(distances, filenames)
              
        results[expected_location]["total"] += 1
        
        preds["actual"].append(expected_location)
        preds["pred"].append(predicted_location)
                
        if (predicted_location == expected_location):
            results["overall"] += 1
            results[predicted_location]["correct"] += 1

    get_confusion_matrix(preds)
    print_results(results)
    

