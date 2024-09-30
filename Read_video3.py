# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import speake3
import pytesseract
import pyttsx3
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
but = 2
GPIO.setup(but,GPIO.IN)

engine = speake3.Speake() # Initialize the speake engine
engine.set('voice', 'en')
engine.set('speed', '150')
engine.set('pitch', '50')
engine.say("WELCOME") #String to be spoken
engine.talkback()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = 'coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)

# initialize a list of colors to represent each possible class label
np.random.seed(42)

COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

# derive the paths to the YOLO weights and model configuration
weightsPath = 'yolov3.weights'
configPath = 'yolov3.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions


(W, H) = (None, None)
vs = cv2.VideoCapture(0)
time.sleep(2)
while True:
        
        # read the next frame from the file

        (grabbed, frame) = vs.read()
        cv2.imshow('input',frame)
        cv2.waitKey(1)
        
        if(GPIO.input(but)==0):
                tm=0
                while(GPIO.input(but)==0):
                        tm=tm+1
                        time.sleep(0.3)
                        print(tm)
                if(tm<5):
                
                        print('OBJECT DETECTION:')


                        if W is None or H is None:
                                (H, W) = frame.shape[:2]
                        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                swapRB=True, crop=False)
                        net.setInput(blob)
                        layerOutputs = net.forward(ln)
                        boxes = []
                        confidences = []
                        classIDs = []

                        for output in layerOutputs:
                                # loop over each of the detections
                                for detection in output:
                                        # extract the class ID and confidence (i.e., probability)
                                        # of the current object detection
                                        scores = detection[5:]
                                        classID = np.argmax(scores)
                                        confidence = scores[classID]

                                        if confidence > args["confidence"]:

                                                box = detection[0:4] * np.array([W, H, W, H])
                                                (centerX, centerY, width, height) = box.astype("int")
                                                x = int(centerX - (width / 2))
                                                y = int(centerY - (height / 2))
                                                boxes.append([x, y, int(width), int(height)])
                                                confidences.append(float(confidence))
                                                classIDs.append(classID)

                        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])

                        # ensure at least one detection exists
                        if len(idxs) > 0:
                                # loop over the indexes we are keeping
                                for i in idxs.flatten():
                                        # extract the bounding box coordinates
                                        (x, y) = (boxes[i][0], boxes[i][1])
                                        (w, h) = (boxes[i][2], boxes[i][3])

                                        # draw a bounding box rectangle and label on the frame
                                        color = [int(c) for c in COLORS[classIDs[i]]]
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                                confidences[i])
                                        cv2.putText(frame, text, (x, y - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                        print(LABELS[classIDs[i]])

                                        engine.say(LABELS[classIDs[i]]) #String to be spoken
                                        engine.talkback()
                        cv2.imshow('output',frame)
                        cv2.waitKey(2000)
 
                        time.sleep(2)
                else:
                        print('OCR:  captured image')
                        engine.say("image is captured") #String to be spoken
                        engine.talkback()

                        if not grabbed:
                               break
                       
                        print(pytesseract.image_to_string(frame))
                        engine.say(pytesseract.image_to_string(frame))
                        engine.talkback()
                        
                        cv2.imshow('output',frame)
                        cv2.waitKey(2000)

                        time.sleep(2)
