# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 09:42:19 2025

@author: mitoa
"""
# Edited to use f-string and solve the hang issue (the window does not end properly when run on Spyder6.)

import os
from pathlib import Path
import numpy as np
import cv2 as cv

MIN_NUM_KEYPOINT_MATCHES = 50

def main():
    path1 = Path.cwd() / 'night_1'
    path2 = Path.cwd() / 'night_2'
    path3 = Path.cwd() / 'night_1_registered'
    path3.mkdir(parents=True, exist_ok=True)

    night1_files = sorted([f for f in os.listdir(path1) if f.lower().endswith('.png')])
    night2_files = sorted([f for f in os.listdir(path2) if f.lower().endswith('.png')])             

    for i in range(len(night1_files)):    
        img1 = cv.imread(str(path1 / night1_files[i]), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(str(path2 / night2_files[i]), cv.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        print(f"Displaying and Processing: {night1_files[i]}")

        # 1. Find matches
        kp1, kp2, best_matches = find_best_matches(img1, img2)
        
        # 2. Show the matching lines on screen
        img_match = cv.drawMatches(img1, kp1, img2, kp2, best_matches, outImg=None)
        # --- DRAW THE VERTICAL SEPARATOR LINE ---
        height, width = img1.shape[:2]
        # Using Bright Green (0, 255, 0) and thickness of 3 so it's impossible to miss
        cv.line(img_match, (width, 0), (width, height), (0, 255, 0), 3)
        
        cv.imshow('Keypoint Matches', img_match)
        cv.waitKey(2000)  # Shows the match lines for 2 seconds

        # 3. Register the image
        img1_registered = register_image(img1, img2, kp1, kp2, best_matches)

        # 4. Save (Overwrites existing)
        out_filename = f"{Path(night1_files[i]).stem}_registered.png"
        cv.imwrite(str(path3 / out_filename), img1_registered)

        # 5. Show the "Blink" on screen to check alignment
        # This will toggle between img1_registered and img2
        blink(img1_registered, img2, 'Registration Check (Blink)', num_loops=10)
        
        # Cleanup for Spyder stability
        cv.destroyAllWindows()
        cv.waitKey(1) 
    
    print("Processing complete. All windows closed.")

def find_best_matches(img1, img2):
    orb = cv.ORB_create(nfeatures=1000) 
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    if desc1 is None or desc2 is None:
        return kp1, kp2, []
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desc1, desc2), key=lambda x: x.distance)
    return kp1, kp2, matches[:MIN_NUM_KEYPOINT_MATCHES]

def register_image(img1, img2, kp1, kp2, best_matches):
    if len(best_matches) >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        h_matrix, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        return cv.warpPerspective(img1, h_matrix, (img2.shape[1], img2.shape[0]))
    return img1

def blink(image_1, image_2, window_name, num_loops):
    """Toggles two images in the same window."""
    for _ in range(num_loops):
        cv.imshow(window_name, image_1)
        if cv.waitKey(400) & 0xFF == ord('q'): break # Press 'q' to skip to next image
        cv.imshow(window_name, image_2)
        if cv.waitKey(400) & 0xFF == ord('q'): break
    cv.destroyWindow(window_name)

if __name__ == '__main__':
    main()
