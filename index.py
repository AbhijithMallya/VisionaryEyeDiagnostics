from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
image_size = (128, 128)

label_mapping = {
    0 :'Cataract',
    1 :'Diabetic Retinopathy',
    2 :'Glaucoma',
    3 :'Healthy Eyes'
}


# Load the trained model
model = load_model('model_saved')

# Load the label encoder classes
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure the 'uploads' folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the uploaded file to the 'uploads' folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load and preprocess the uploaded image
        new_image = cv2.imread(file_path)
        resized_image = cv2.resize(new_image, image_size)
        preprocessed_image = preprocess_input(resized_image.reshape(1, *image_size, 3))

        # Make a prediction
        prediction = model.predict(preprocessed_image)
        predicted_class_index = int(np.argmax(prediction))

        # Convert prediction array to a Python list
        prediction_list = prediction.flatten().tolist()

        # Decode the predicted class
        predicted_class = label_encoder_classes[predicted_class_index]
        
        predicted_class = label_mapping[predicted_class]
        print('*****************************************************')
        print('****************predictewd class : ' , predicted_class)
        #return jsonify({'result': predicted_class, 'prediction': prediction_list})
        return render_template('index.html',span = predicted_class)

    return render_template('index.html',span = predicted_class)

# if __name__ == '__main__':
#     app.run(debug=True)

