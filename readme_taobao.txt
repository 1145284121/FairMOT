# Object detection & tracking codes modified from: https://github.com/ifzhang/FairMOT
# Pre-trained models saved as: FairMOT/models/fairmot_lite.pth
# Sample videos saved in: FairMOT/All_Data/*
# To process a single video: 
## cd FairMOT/src
## python demo.py mot --load_model ../models/fairmot_lite.pth --input-video ../All_Data/video1/video1.mp4 --output-root ../All_Data/video1
## A window is prompted to show the detection results on-the-fly, which can be stopped by pressing ESc key
## Afterwards a folder 'frame' is created in output-root to store the detected frames, and a video result.mp4 is generated by joining the frames (with ffmpeg)

# opencv is used to load video (class LoadVideo in FairMOT/src/lib/datasets/dataset/jde.py)
# Main code starts from FairMOT/src/track.py, FairMOT/src/lib/tracker/multitracker.py
# Frame visualization markup in FairMOT/src/lib/tracking_utils/visualization.py