import glob, os
import numpy as np
from time_comp import run_mp
from holistic import run_hol
import pandas as pd

files = []
os.chdir("/home/james/neuro/Data/hf")
for file in glob.glob("*.mp4"):
    print(file)
    files.append(file)
print(files)


#compute times in mediapipe
times_mp=pd.DataFrame(columns=files)
print(times)
for i in range(10):
    for file in files :
        times.append()
        #for i in range(10) :
        time = run_mp(file)
        print(f'Time of the video {file}: {time} [s].' )
        times.append(time)

#compute times in holistic
times_hol=pd.DataFrame(columns=files)
print(times_hol)
for i in range(10):
    for file in files :
        times.append()
        #for i in range(10) :
        time = run_hol(file)
        print(f'Time of the video {file}: {time} [s].' )
        times_hol.append(time)
