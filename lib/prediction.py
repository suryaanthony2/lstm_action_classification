import numpy as np
import pathlib
import os

def classify(input_x, input_y, input_z, model):
    path = str(pathlib.Path().resolve()) + "\\video\\train"

    classes = os.listdir(path)

    prediction = model.predict([input_x, input_y, input_z], verbose=0)
    prediction_result = np.argmax(prediction)
    
    prediction_prob = prediction[:, prediction_result]
    
    if prediction_result == 0:
        return classes[0], prediction_prob
    elif prediction_result == 1:
        return classes[1], prediction_prob
    elif prediction_result == 2:
        return classes[2], prediction_prob
    elif prediction_result == 3:
        return classes[3], prediction_prob
    elif prediction_result == 4:
        return classes[4], prediction_prob