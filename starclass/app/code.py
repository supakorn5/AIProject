import numpy as np

Star = {
    0: "chanel3",
    1: "chanel7"
}

def Predict_star(model, hog1):
    # Ensure 'hog1' is a numerical array with at least one feature before using it for prediction
    if len(hog1) == 0:
        raise ValueError("Input data should contain at least one feature.")

    predictions = model.predict(np.array(hog1).reshape(1, -1))
    return Star[predictions[0]]
