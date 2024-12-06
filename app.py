import os
from flask import Flask, request, render_template
from src.predict import predict_rock_type

app = Flask(__name__)

# Define the upload folder path
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the uploads directory if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
