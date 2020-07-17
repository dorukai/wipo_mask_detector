
import imutils
import time
import cv2
import os
import detect_srv
from imutils.video import WebcamVideoStream

class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video =  WebcamVideoStream(src=1).start()

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self, faceNet, maskNet, confidence):
        # extracting frames
        frame = self.video.read()
        print("Frame => ",frame)

        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_srv.detect_and_predict_mask(frame, faceNet, maskNet, confidence_val=confidence)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Maske Var!" if mask > withoutMask else "Maske Yok!"
            color = (0, 255, 0) if label == "Maske Var!" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


