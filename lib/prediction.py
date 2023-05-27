import numpy as np

def classify(input_x, input_y, input_z, model):
    prediction = model.predict([input_x, input_y, input_z])
    prediction_result = np.argmax(prediction)
    
    if prediction_result == 0:
        return "Age-Uke"
    elif prediction_result == 1:
        return "Gedan-Barai"
    elif prediction_result == 2:
        return "Mawashi-Uchi"
    elif prediction_result == 3:
        return "Oi-Zuki"
    elif prediction_result == 4:
        return "Shuto-Uke"