import io
from flask import Flask, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
import os
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

print('app.py file')
@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

MODEL_PATH = 'Models/VGG19_weights.h5'

@app.route('/predict', methods = ['GET', 'POST'])
def infer_image():
    print('redirected to /predict ')
    # Load the saved model
    model = load_model(MODEL_PATH)

    # Get the uploaded image from the request
    img = request.files['file'].read()

   # Convert the image to a numpy array
    img_array = np.array(Image.open(io.BytesIO(img)).resize((224, 224)))

    # Preprocess the image
    img_array = img_array / 255.0

    # Make predictions
    predictions = model.predict(np.expand_dims(img_array, axis=0))

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])

    print("pred ",str(predictions))
    # Return the predicted class as a response
    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)