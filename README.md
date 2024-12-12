# Player Tracking Evaluation with YOLOv8, ByteTrack, and BoT-SORT

## Overview

This project evaluates the performance of different tracking algorithms for player tracking in volleyball match videos. The focus is on comparing the YOLOv8 Default Tracker, ByteTrack, and BoT-SORT in terms of tracking accuracy, identity consistency, and robustness.

## Trackers Evaluated

- **YOLOv8 Default Tracker**: Baseline tracker using YOLOv8’s default tracking mechanism.
- **ByteTrack**: Combines strong detection with a simple association mechanism to improve tracking accuracy.
- **BoT-SORT**: Enhances tracking with appearance features and advanced motion models for improved identity preservation.

## Procedure

### 1. **Dataset Preparation**
   - Volleyball match videos featuring multiple players.
   - Ensured videos captured complex player interactions to test tracker robustness.

### 2. **Detection and Tracking**
   - **YOLOv8** was used for object detection across all trackers.
   - Each tracker (YOLOv8 Default, ByteTrack, BoT-SORT) was applied to the videos.
   - Player bounding boxes and identity assignments were generated for each frame.

### 3. **Qualitative Analysis**
   - **BoT-SORT Tracker**:  
     - Maintained consistent IDs and bounding boxes.
   - **YOLOv8 Tracker**:  
     - Efficient but exhibited lower confidence scores.
   - **ByteTrack**:  
     - Accurate bounding boxes and high confidence scores.
   - **BoT-SORT**:  
     - Best at maintaining player identities due to appearance features and advanced motion models.

### 4. **Challenges Noted**
   - **Identity Switches**:  
     - Reduced in BoT-SORT but still present during complex interactions.
   - **Computational Load**:  
     - Advanced trackers like BoT-SORT require more resources, impacting real-time feasibility.
   - **False Positives**:  
     - Low-confidence detections led to occasional tracking of non-player objects.  

## Results Summary

- **BoT-SORT**: Best performance in maintaining player identities.
- **ByteTrack**: Accurate bounding boxes with fewer identity switches.
- **YOLOv8 Default Tracker**: Efficient but less accurate in maintaining consistent IDs.

## Future Improvements

- Optimize BoT-SORT for real-time performance.
- Reduce false positives by refining confidence thresholds.
- Further minimize identity switches during complex interactions.
