from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np
from time import sleep

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
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = FileVideoStream("/home/james/neuro/Data/test.mp4").start()
#vs = VideoStream(src=0).start()
vs = FileVideoStream('/home/james/neuro/Data/hf/james_walk.mp4').start()

sleep(2.0)
fps = FPS().start()

check = 0
speed = 15
sf = 30
t = 0.4
tf = 0.8

def follow():
	#cv2.waitKey(2000)
	if xcen < 150:
		#eh.motor.two.forward(speed)
		#sleep(t)
		#eh.motor.two.stop()
		print("Left: xcen = ",xcen)

	elif xcen > 250:
		#eh.motor.one.forward(speed)
		#sleep(t)
		print("Right: xcen = ",xcen)
		#eh.motor.one.stop()

	elif startX < 50 and endX > 320:
		stop()
		print("Object chase complete!!")
	elif xcen == 0:
		stop()
	else:
		#eh.motor.forwards(sf)
		#sleep(tf)
		#eh.motor.stop()
		print("Forward: xcen = ",xcen)


def stop():
	#continue
	#eh.motor.stop()
	#print("Stopped following")
	return

# loop over the frames from the video stream
j=0
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    cam = vs.read()
    frame = cv2.flip(cam,1)
    #if frame is None :
    #    break
    frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),	0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

	# loop over the detections
    for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in IGNORE:
                continue
			#compute the (x, y)-coordinates of the bounding box for
			# the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(464, 0, 600, 760) #(238, 0, 400, 380)

            plt.scatter(j, 100/np.log(np.mean([endX-startX, endY-startY])), c='red')
            plt.xlabel('Timepoint')
            plt.ylabel('Relative distance from camera')
            plt.pause(0.00000000000000005)
            j+=1

            xcen = (endX + startX)/2
            ycen = (endY + startY)/2
			# draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            check = 1
        else:
            check = 0
            xcen = 0

	# show the output frame
    winname = "Frame"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 0,0)
    #cv2.resizeWindow(winname, 1000000,1000000)
    cv2.imshow(winname, frame)
    key = cv2.waitKey(1) & 0xFF

    if check == 1:
        follow()
    else:
        stop()
        #print("Stopped following")

    #if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

	# update the FPS counter
    fps.update()

# stop the timer and display FPS informatio
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
#cv2.destroyAllWindows()
plt.show()
vs.stop()
