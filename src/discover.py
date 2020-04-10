#!/usr/bin/env python

import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import feature_extraction as ft
import sys
import image_search
import os.path
import progressbar

# global variables
# vocabularies
sift_vocabulary = None
colorhist_features = None
# results
sift_candidates = None
colorhist_candidates = None

# Command line parsing is handled by the ArgumentParser object
parser = argparse.ArgumentParser(description="Query tool to query the database created by the database tool (dbt.py). Retrieve images based on image content and metadata.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("database_colorhist", help="Path to the database of the colorhist features")
parser.add_argument("database_sift", help="Path to the database of the sift features")
parser.add_argument("query", help="Query Video")

args = parser.parse_args()

# Get database locations
base_colorhist = os.path.splitext(args.database_colorhist)[0]
base_sift = os.path.splitext(args.database_sift)[0]

def get_location(loc):
        if loc == "nk":
            return "Nieuwe Kerk"
        if loc == "rh":
            return "Stadhius"
        if loc == "oj":
            return "Oude Jan"
        else:
            return "Could not match any location"
            
def update_results(loc, d):
    if loc == "nk":
        locations_counts['nk'] += 1/d
    if loc == "rh":
        locations_counts['rh'] += 1/d
    if loc == "oj":
        locations_counts['oj'] += 1/d

# Starting point of the script
# =======================================

if __name__ == '__main__':
   
    print '\nDelft Discover Query Tool'
    print '================================\n'
    print "Query with " + args.query
    
    db_name_colohist = args.database_colorhist
    search_colorhist = image_search.Searcher(db_name_colohist)
    
    db_name_sift = args.database_sift
    search_sift = image_search.Searcher(db_name_sift)
    
    # Switch to True to see the classification of each frame
    DEBUG = False
    
    cap = cv2.VideoCapture(args.query)
    locations_counts = {
        "nk" : 0,
        "rh" : 0,
        "oj" : 0
    }
    
    # Load vocabularies
    # SIFT
    fname_sift = base_sift + '_sift_vocabulary.pkl'
    # Load the vocabulary to project the features of our query image on
    with open(fname_sift, 'rb') as f:
        sift_vocabulary = pickle.load(f)

    # Color Histogram
    fname_colorhist = base_colorhist + '_colorhist.pkl'
    # Load all colorhistogram features of our training data
    with open(fname_colorhist, 'rb') as f:
        colorhist_features = pickle.load(f)
    
    frames = []
    
    # Save all the frames in the video
    while(True):
        ret, frame = cap.read()
        frames.append(frame)
        if frame is None:
            break;
    
    # Take the middle third of the frames
    frames = frames[len(frames)//3:len(frames)//3*2]

    # Set up progress bar
    total = len(frames)
    bar = progressbar.ProgressBar(maxval=total, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    print 'Geo-Prediction Detection:'
    bar.start()
    count = 0

    # For each of the frames
    for query in frames:

        sift_query = ft.get_sift_features_frame(query)
        # Get a histogram of visual words for the query image
        image_words = sift_vocabulary.project(sift_query)
        # Use the histogram to search the database
        sift_candidates = search_sift.query_iw('sift', image_words)

        # Get colorhistogram for the query image
        colorhist_query = ft.get_colorhist_frame(query)
        # Compare the query colorhist with the dataset and retrieve an ordered list of candidates
        colorhist_candidates = search_colorhist.candidates_from_colorhist(colorhist_query, colorhist_features, cosine_similarity = True)
   
        # If candidates exists, show the top N candidates
        N = 1
        
        if not colorhist_candidates == None:
            distances = colorhist_candidates[1][0:N]
            filenames = colorhist_candidates[0][0:N]
            
            # Check if the match is good enough
            if (distances[0] < 0.175):
                update_results(filenames[0][6:8], distances[0])
                if DEBUG:
                    print("Location detected by Color hist: " + get_location(filenames[0][6:8]))
            # Proceed to SIFT if the match is not good enough
            else:
                if not sift_candidates == None:
                    sift_winners = [search_sift.get_filename(cand[1]) for cand in sift_candidates][0:N]
                    sift_distances = [cand[0] for cand in sift_candidates][0:N]
                    
                    update_results(sift_winners[0][29:31], sift_distances[0]/100)
                    if DEBUG:
                        print("Location detected by SIFT: " + get_location(sift_winners[0][29:31]))  
        
        bar.update(count)
        count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    bar.finish()
    
    res = max_key = max(locations_counts, key=locations_counts.get)
    print 'The location of the video is ' + get_location(res)
    
