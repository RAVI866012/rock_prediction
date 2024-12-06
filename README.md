1. Dataset Preparation:
You need a dataset containing images of rocks labeled with their respective types (e.g., sedimentary, igneous, metamorphic). You can either:

Use a publicly available dataset (e.g., Kaggle has rock datasets such as the Rock Type Classification Dataset or Geology Dataset).
Collect your own dataset by taking pictures of rocks and labeling them.


2. Project Structure:
Here’s a suggested folder structure for your project:


rock_classification/
│
├── data/
│   ├── train/
│   │   ├── sedimentary/
│   │   ├── igneous/
│   │   └── metamorphic/
│   ├── test/
│   └── README.md
│
├── models/
│   └── rock_model.h5
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── app.py





3. Environment Setup:
Create a requirements.txt file to install the necessary libraries:


Copy code
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
flask
opencv-python
Pillow


Install these dependencies by running:

command>

pip install -r requirements.txt



4. Data Preprocessing:
In src/data_preprocessing.py, write code for loading and preprocessing the images, including resizing and normalization:


1 src/data_preprocessing.py>

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_images(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    label_map = {'sedimentary': 0, 'igneous': 1, 'metamorphic': 2}  # Example label map
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    img_path = os.path.join(folder_path, image_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label_map[folder])
    
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(label_map))  # One-hot encode labels
    return images, labels

def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)


    
5. Model Building:


Create a Convolutional Neural Network (CNN) model for rock type classification in src/model.py:



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: sedimentary, igneous, metamorphic
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    
6. Model Training:

   
In src/train.py, write the code to train your model:


src/train.py>

from src.data_preprocessing import load_images, split_data
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model():
    images, labels = load_images('data/train')
    X_train, X_test, y_train, y_test = split_data(images, labels)
    
    model = create_model()
    checkpoint = ModelCheckpoint('models/rock_model.h5', save_best_only=True, monitor='val_loss')
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[checkpoint])
    
    model.save('models/rock_model.h5')

if __name__ == '__main__':
    train_model()


    
7. Making Predictions:

   
Create a predict.py file to load the trained model and make predictions on new images:


predict.py>

from tensorflow.keras.models import load_model
import cv2
import numpy as np

def predict_rock_type(image_path):
    model = load_model('models/rock_model.h5')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and expand dims for batch prediction
    prediction = model.predict(img)
    classes = ['sedimentary', 'igneous', 'metamorphic']
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

    
8. Web Application with Flask:


To make predictions via a web interface, use Flask. In app.py, write the following code:



from flask import Flask, request, render_template
from src.predict import predict_rock_type

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    image_path = f'./uploads/{file.filename}'
    file.save(image_path)
    
    result = predict_rock_type(image_path)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


    
9. HTML Templates:



Create simple HTML templates to upload images and display results.

templates/index.html:



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Rock Type Prediction</title>
</head>
<body>
    <h1>Upload a Rock Image</h1>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <button type="submit">Predict</button>
    </form>
</body>
</html>


templates/result.html:



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediction Result</title>
</head>
<body>
    <h1>Rock Type Prediction</h1>
    <p>The predicted rock type is: {{ result }}</p>
    <a href="/">Go Back</a>
</body>
</html>



10. Running the Application:

    
Start the Flask app with:


11. Conclusion:
12. 
Now, when you visit http://127.0.0.1:5000/,
 you can upload rock images, and the model will predict the rock type (sedimentary, igneous, or metamorphic).
You can further enhance this project by fine-tuning the model, adding more rock types, or improving the user interface.






To train the rock-type classification model, follow these steps. Make sure you've already set up the project structure, including the dataset, model, and preprocessing code as outlined earlier. Here are the commands and instructions:

1. Set Up Your Environment:
If you haven't already installed the dependencies, run the following command to install the required libraries from the requirements.txt file:


pip install -r requirements.txt


2. Prepare Your Dataset:

Ensure that your dataset is organized properly. It should be placed in the data/train/ directory with subfolders for each class (e.g., sedimentary/, igneous/, metamorphic/), containing the respective rock images. Also, if you're using a test set, place the test images in the data/test/ folder.

Example:


rock_classification/
│
├── data/
│   ├── train/
│   │   ├── sedimentary/
│   │   ├── igneous/
│   │   └── metamorphic/
│   ├── test/
│   └── README.md


3. Train the Model:
You can train the model by running the training script. Make sure you are in the project’s root directory, then execute the following command:


python src/train.py

This command will:

Load the images from the data/train/ directory.

Preprocess the images (resize, normalize, etc.).

Split the data into training and validation sets.


Train the model and save the best version to models/rock_model.h5.

4. Monitor Training:
   
The model training process will output the training progress (e.g., loss and accuracy). You can adjust the number of epochs or other parameters in the train.py script as needed to optimize performance.

You can modify the number of epochs and batch size in the model.fit function inside train.py:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[checkpoint])
Epochs: The number of times the model will iterate over the entire dataset.
Batch size: The number of images used in one training step.

5. Model Saving:
   
After the model is trained, the best weights are saved as rock_model.h5 inside the models/ directory. You can use this model for predictions or further tuning.

7. Verify the Model:
   
To check the performance of the model after training, you can load the saved model (rock_model.h5) and make predictions. You can either:

Use the predict.py script to test the model on individual images.
Use a web interface (Flask app) as discussed earlier.


