import glob, os
import numpy as np
import pandas as pd
import cv2
import subprocess

#function to get length of a mp4 video
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def test_times() :
    files = []
    os.chdir("/home/james/neuro/Data/hf")
    for file in glob.glob("*.mp4"):
        print(file)
        files.append(file)
    print(files)

    lengths = [get_length(filename) for filename in files]

    from mp.time_comp import run_mp
    #compute times in mediapipe
    times_mp=pd.DataFrame(columns=files)
    for i in range(10):
        time_iter = []
        for file in files :
            #for i in range(10) :
            time = run_mp(file)
            print(f'Time of the video {file}: {time} [s].' )
            time_iter.append(time)
        times_mp.loc[times_mp.shape[0]] = time_iter
        print(times_mp)

    """
    from mp.holistic import run_hol
    #compute times in mediapipe
    times_hol=pd.DataFrame(columns=files)
    for i in range(2):
        time_iter = []
        for file in files :
            #for i in range(10) :
            time = run_hol(file)
            print(f'Time of the video {file}: {time} [s].' )
            time_iter.append(time)
        times_hol.loc[times_hol.shape[0]] = time_iter
        print(times_hol)

    cmd = '. /home/james/anaconda3/etc/profile.d/conda.sh && conda deactivate'
    subprocess.call(cmd, shell=True, executable='/bin/bash')
    cmd = '. /home/james/anaconda3/etc/profile.d/conda.sh && conda list'
    subprocess.call(cmd, shell=True, executable='/bin/bash')
    cmd = '. /home/james/anaconda3/etc/profile.d/conda.sh && conda activate human'
    subprocess.call(cmd, shell=True, executable='/bin/bash')
    """
    from human.time_comp import human_time
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
    net = cv2.dnn.readNetFromCaffe("/home/james/neuro/scripts/human/MobileNetSSD_deploy.prototxt.txt", "/home/james/neuro/scripts/human/MobileNetSSD_deploy.caffemodel")

    #compute times in human
    times_h=pd.DataFrame(columns=files)
    for i in range(10):
        time_iter = []
        for file in files :
            #for i in range(10) :
            time = human_time(file, net, CLASSES, IGNORE, COLORS)
            print(f'Time of the video {file}: {time} [s].' )
            time_iter.append(time)
        times_h.loc[times_h.shape[0]] = time_iter
        print(times_h)

    return lengths, times_mp, times_h

#lengths, times_mp, times_hol, times_h = test_times()
#times_mp.to_csv('/home/james/neuro/Data/times_mp.csv', index=False)
#times_hol.to_csv('/home/james/neuro/Data/times_hol.csv', index=False)
#times_h.to_csv('/home/james/neuro/Data/times_h.csv', index=False)
