# volleyball_tracker

This project is realised for the course of ML4CV at UniBo. 

The aim is to build an AI system able to track the volleyball players inside videos; the videos are in reference of my volleyball team and the camera is always in the same spot behind the players and the system will track the players in the close half of the field.

## Project's details
The videos are given in format .m4v that is not suitable for OpenCV and other CV operations, so it was necessary to convert those in .mp4 thanks to the convert_videos.sh 
```bash
./convert_videos.sh
```

## Object detection
### Folder with images from the video
First of all, I need to perform object detection on the frames of the input video. Obviously, this operation can't be done directly on the video file so I have to extract some frames; this was achieved with some basic operations of OpenCV and Numpy libraries.
From a single video, I extract 1000 random frames (I can fix this parameter later based on my results) and these are inserted inside a new folder.

### Yolov3
I choose to perform the detection with Yolov3, a popular pretrained detector that can give good results. 
To use it, I had to download yolov3.weights, yolov3.cfg and coco.names
