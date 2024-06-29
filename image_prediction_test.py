from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
import cv2
MODEL_PATH = 'Models/VGG19_weights.h5'

def isFileExist(path):
    if os.path.exists(path):
        print('File exists')
    else:
        print('File does not exist')
        
def listDir(path):
    if os.path.exists(path):
        print('File exists')
    else:
        print('File does not exist')

model = load_model(MODEL_PATH)
# model.summary()
app = Flask(__name__)
TRAIN_IMAGE_PATH = '../Steel Defect Detection Dataset/train_images/0a1cade03.jpg'

isFileExist(TRAIN_IMAGE_PATH)
# listDir()
def model_predict(img_path, model):
    print('model_predict invoked')
    img = image.load_img(img_path, target_size = (224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis = 0)
    preds = model.predict(x)
    actual_prediction = np.argmax(preds) + 1
    return actual_prediction

random_output = model_predict(TRAIN_IMAGE_PATH, model)

print("test wala file")
print(random_output)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("uploading method called")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        # return result
        print("-> ",file_path)
        print('types resullt is ',type(result), ' value isnside -> ',result)
        # x =  overlay_defected_area(file_path,model )
        # print('x = ',x)
        # return result)
        return "hello"
    return None

def overlay_defected_area(img_path, model):
    # Load the image
    img = cv2.imread(img_path)
    img_copy = img.copy()

    # Predict the defected area
    defected_area = model_predict(img_path, model)

    # Overlay the defected area with red color
    img_copy[defected_area == 1] = (0, 0, 255) # BGR color code for red

    # Save the output image
    output_path = img_path.split('.')[0] + '_overlayed.jpg'
    cv2.imwrite(output_path, img_copy)

    # Return the path of the output image
    return output_path

def overlay_defects(img_path, prediction):
    # Load the image
    img = cv2.imread(img_path)

    # Create a mask with the defected area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[prediction == 1] = 255

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Overlay the defected areas with a red color
    img[mask != 0] = [0, 0, 255]

    # Combine the masked image and the original image
    combined_img = cv2.addWeighted(masked_img, 0.5, img, 0.5, 0)

    # Save the output image
    output_path = img_path.replace(".jpg", "_defects.jpg")
    cv2.imwrite(output_path, combined_img)

    # Return the output image path
    return output_path
if __name__ == '__main__':
    app.run(debug=True)