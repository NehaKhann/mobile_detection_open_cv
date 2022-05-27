from YOLOTorchInference import Inference

"""
Run the script in the same path as the weights or put the correct file path

Arguments to set before running inference:

1. weightPath: path to the weights of the model to be used. SELF EXPLANATORY NAAM HAI
2. captureMode: (cant think of a better name for the parameter right now)
    a. Send int 0 for webcam.
    b. Send a string containing the path to the video
"""

weightPath = './mobile_19_may_2022.pt'
captureMode = 0

mobileDetector = Inference(weightPath, captureMode)
mobileDetector()
#mobileDetector.detect()
#mobileDetector.display_frame()
