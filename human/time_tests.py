import glob, os
import numpy as np
import cv2
from time_comp import human_time

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# Ignore all objects in the frame except Person
IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

files = []
os.chdir("/home/james/neuro/Data/hf")
for file in glob.glob("*.mp4"):
    print(file)
    files.append(file)
print(files)


#compute times in mediapipe
times=[]
for file in files :
    #for i in range(10) :
    time = human_time(file, net, CLASSES, IGNORE, COLORS)
    print(f'Time of the video {file}: {time} [s].' )
    times.append(time)

tot_time = np.sum(times)
print(tot_time)
