# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:24:23 2026
Update on transient_detector. 
This version uses f-strings, zip, 
and solved hang issues.
Built on Spyder6
January 16, 2026
@author: Mito Akiyoshi
"""

import cv2 as cv
from pathlib import Path

# Configuration
PAD = 5 
THRESHOLD_VAL = 30 
NUM_TRANSIENTS = 2 

def find_transient(image_to_label, diff_image, pad):
    """Finds brightest spot, labels it, and erases it from the search image."""
    height, width = diff_image.shape
    _, maxVal, _, maxLoc = cv.minMaxLoc(diff_image)
    
    if (pad < maxLoc[0] < width - pad and 
        pad < maxLoc[1] < height - pad and 
        maxVal > THRESHOLD_VAL):
        
        # Draw the target circle on the display image (White outline)
        cv.circle(image_to_label, maxLoc, 10, 255, 1)
        # Eraser: Black out this area in the diff_image so we don't find it again
        cv.circle(diff_image, maxLoc, 10, 0, -1)
        return True
    
    return False

def main():
    path1 = Path.cwd() / 'night_1_registered_transients'
    path2 = Path.cwd() / 'night_2'
    out_path = Path.cwd() / 'night_1_2_transients'
    out_path.mkdir(exist_ok=True)

    # Use glob to get files and sort them to ensure they match
    night1_files = sorted(path1.glob('*.png'))
    night2_files = sorted(path2.glob('*.png'))

    for p1, p2 in zip(night1_files, night2_files):
        img1 = cv.imread(str(p1), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(str(p2), cv.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            continue

        # 1. Show raw difference for 2 seconds
        diff_imgs1_2 = cv.absdiff(img1, img2)
        cv.imshow('Difference', diff_imgs1_2)
        cv.waitKey(2000) 
        
        # 2. Detection (Working on a copy of the difference)
        working_diff = diff_imgs1_2.copy() 
        detections_found = 0
        for _ in range(NUM_TRANSIENTS):
            if find_transient(img1, working_diff, PAD):
                detections_found += 1

        # 3. Handle Detections
        if detections_found > 0:
            # Console Log using f-strings
            print(f"\nTRANSIENT DETECTED between {p1.name} and {p2.name}\n")
            
            # Text labels on the image
            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(img1, p1.name, (10, 25), font, 1, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(img1, p2.name, (10, 55), font, 1, (255, 255, 255), 1, cv.LINE_AA)
            
            # Create Blended Survey Image
            blended = cv.addWeighted(img1, 1, diff_imgs1_2, 1, 0)
            
            # Display result in the 'Surveyed' window
            cv.imshow('Surveyed', blended)
            cv.imwrite(str(out_path / f"{p1.stem}_DETECTED.png"), blended)
            
            # Pause to show results; press 'q' to quit early
            if cv.waitKey(2500) & 0xFF == ord('q'):
                break
        else:
            print(f"\nNo transient detected between {p1.name} and {p2.name}\n")

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
