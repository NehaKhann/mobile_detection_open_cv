import torch
import cv2
import numpy as np
import os
import filetype
from datetime import datetime
import pika

class Inference:

  """
  webcam and video dono
  """

  def __init__(self, weightPath, captureMode):

    self.model = self.load_model(weightPath)
    self.captureMode = captureMode
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.classes = self.model.names
    self.model.conf = 0.5
    self.timeToDetect = []
    
    self.model.to(self.device)


  def __call__(self):
    
    if self.captureMode in [0, 1] or filetype.is_video(self.captureMode):
      frames = 0
      gst_str = ('v4l2src device=/dev/video{} ! image/jpeg, width={}, height={},framerate=30/1 ! jpegdec ! video/x-raw, framerate=30/1 ! videoconvert ! appsink ').format(0, 1920, 1080)
      cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

      assert cap.isOpened()
        
      while True:
        
          ret, frame = cap.read()
          frames += 1
          if ret == True: 
            results = self.detect(frame)
            frame = self.plot_boxes(results, frame)
          else:
            break

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    
      cap.release()
      print(frames, len(self.timeToDetect))
    elif filetype.is_image(self.captureMode):
      img = cv2.imread(self.captureMode)
      results = self.detect(img)
      self.display_results(results)
      #img = self.plot_boxes(results, img)
      cv2.waitKey(0)
    # print('da time of all da detections: ', self.timeToDetect)
    # print('mean of all da times: ', np.mean(self.timeToDetect))
    
#     cap.release()
    cv2.destroyAllWindows()


  def load_model(self, weightPath):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = weightPath) 
    return model
  

  def detect(self, frame):
    frame = frame[..., ::-1]
    start_time = datetime.now()
    results = self.model(frame)
    end_time = datetime.now()
#     print(results.xyxyn[0][:])
    labels, cord, conf = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxyn[0][:, -2].cpu().numpy()
    if len(labels) > 0:
      self.timeToDetect.append((end_time-start_time).total_seconds())
    return labels, cord, conf
  

  def class_to_label(self, index):
      return self.classes[int(index)]
  

  def plot_boxes(self, results, frame):
    labels, cord, conf = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, self.class_to_label(labels[i]) + " " + str(conf[0]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        print("Mobile")
#         cv2.putText(frame, str(conf), (x1 + 5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    
    print("Frame procedd")
    #cv2.imshow('YOLOv5 Detection', frame)
    #cv2_imshow(frame)
    return frame
  

  def display_results(self, results):
    results.print()
  

  def display_frame(self, frame):
    cv2.imshow('YOLOV5 Detection', frame)
