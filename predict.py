from tensorflow.keras.models import load_model
import cv2
import numpy as np

def predict_rock_type(image_path):
    model = load_model('models/rock_model.keras')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and expand dims for batch prediction
    prediction = model.predict(img)
    classes = ['sedimentary', 'igneous', 'metamorphic']
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class
