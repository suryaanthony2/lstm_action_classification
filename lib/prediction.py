import numpy as np

def classify(input_x, input_y, input_z, model):
    prediction = model.predict([input_x, input_y, input_z], verbose=0)
    prediction_result = np.argmax(prediction)
    
    prediction_prob = prediction[:, prediction_result]
    
    if prediction_result == 0:
        return "Age-Uke", prediction_prob
    elif prediction_result == 1:
        return "Gedan-Barai", prediction_prob
    elif prediction_result == 2:
        return "Mawashi-Uchi", prediction_prob
    elif prediction_result == 3:
        return "Oi-Zuki", prediction_prob
    elif prediction_result == 4:
        return "Shuto-Uke", prediction_prob