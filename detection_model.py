import numpy as np
import argparse
import cv2

class ObjectDetectionMobileNetModel:
    def __init__(self, prototxtPath, modelPath, confidence):
        # initialize the list of class labels MobileNet SSD was trained to detect
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.confidence = confidence
        self.net = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)

    def process(self, inputImage):
        image = inputImage.copy()
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the neural network
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections and generate a set of bounding box colors for each class
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., the probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
            if confidence > self.confidence:
                # extract the index of the classes label from the 'detections',
                # then compute the (x, y)-coordinates of the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # display the prediction
                label = '{}: {:.2f}%'.format(self.classes[idx], confidence * 100)
                print('[INFO] {}'.format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY), self.colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[idx], 2)
        return image


