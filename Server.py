from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ml_methods import img_preprocessing, img_clahe_preprocessing, CNN_architectures
import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import json

@app.route("/")
def run_app():
    # Render the landing page
    return render_template('landing.html')


@app.route('/classify-single')
def classify_single():
    # Render the single image classification page
    return render_template('classify_single.html')


@app.route('/classify-multiple')
def classify_multiple():
    # Render the multiple image classification page
    return render_template('classify_multiple.html')

@app.route("/upload", methods=["POST"])
def image_upload():
    if request.method == 'POST':
        file = request.files['img']
        if file:
            # Make sure the filename can be use 
            img = secure_filename(file.filename)
            file.save(os.path.join('static/images/', img))
            return render_template('classify_single.html', filename=img)
        else:
            return 'Upload unsuccesfull'


@app.route("/classify/<filename>")
def classify_image(filename):
    file_name = 'data.json'

    # Read JSON data
    with open(file_name, 'r') as file:
        data = json.load(file)
        sel_model = data['model']
        print(sel_model)

    # Load the uploaded image
    img_load = cv.imread('static/images/' + filename)

    # Array declaration
    test_img_3ch = np.zeros((1, 180, 408, 3))
    test_img = np.zeros((1, 180, 408))

    # Load selected model
    model = tf.keras.models.load_model(sel_model)

    # Image preprocessing depending on the selected model
    if sel_model == 'CNN_96_02.keras' or sel_model == 'CNN_98_01_cGAN.keras':
        img = img_clahe_preprocessing(img_load)
        test_img_3ch[0] = img
        pred = model.predict(test_img_3ch)
    else:
        img = img_preprocessing(img_load)
        test_img[0] = img
        pred = model.predict(test_img)

    # Trust calculation
    if pred > 0.5:
        trust = round(float((pred - 0.5) * 2), 4)
        trust_perc = format(trust * 100, '.2f')
        result = 'Positivo'
    else:
        trust = round(float((pred - 0.5) * (-2)), 4)
        trust_perc = format(trust * 100, '.2f')
        result = 'Negativo'
    
    return render_template('classify_single.html', filename=filename, result=result, pred=str(trust),pred_perc=str(trust_perc),model=sel_model)


@app.route("/select-model-single", methods=['POST'])
def select_model_s():
    # Receive the data from the AJAX request
    data = request.get_json()
    
    selected_model = data['model']
    file_name = 'data.json'
    
    # Write the data on the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file)

    return render_template('classify_single.html', model=selected_model)

@app.route("/select-model-multiple", methods=['POST'])
def select_model_m():
    # Receive the data from the AJAX request
    data = request.get_json()
    
    selected_model = data['model']
    
    file_name = 'data.json'
    # Write the data on the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file)

    return render_template('classify_multiple.html', model=selected_model)

@app.route('/upload-images-classify', methods=['POST'])
def upload_files():
    folder_path = 'static/images/'

    # Remove any previous file on the path
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
    
    # Request the list of images from the frontend
    uploaded_imgs = request.files.getlist("img")

    # Array declaration
    test_img = np.zeros((len(uploaded_imgs), 180, 408))
    test_img_3ch = np.zeros((len(uploaded_imgs), 180, 408, 3))

    i = 0
    j = 0
    img_name = []

    file_name = 'data.json'

    # Read JSON data
    with open(file_name, 'r') as file:
        data = json.load(file)
        sel_model = data['model']
        print(sel_model)

    # Save the uploaded images and preprocess according to the correct model
    for img in uploaded_imgs:
        image = secure_filename(img.filename)
        img_name.append(image)
        img.save(os.path.join('static/images/', image))
        img_load = cv.imread('static/images/' + image)
        if sel_model == 'CNN_96_02.keras' or sel_model == 'CNN_98_01_cGAN.keras':
            img = img_clahe_preprocessing(img_load)
            test_img_3ch[i] = img
        else:
            img = img_preprocessing(img_load)
            test_img[i] = img
        i = i + 1

    # Load model
    model = tf.keras.models.load_model(sel_model)

    # Predict the result of the array used before
    if np.all(test_img == 0):
        preds = model.predict(test_img_3ch)
    else:
        preds = model.predict(test_img)

    trust = np.zeros(len(preds))
    trust_percs = np.zeros(len(preds))
    results = []

    # Convert the results from probability to trust
    for p in preds:
        # If a image is positive that means the result varies on the interval [0.5,1]. For the trust result that interval is normalize to [0,1]
        if p > 0.5:
            trust[j] = round(float((p - 0.5) * 2),
                             4)
            trust_percs[j] = format(trust[j] * 100, '.2f')
            results.append('Positivo')
        # If a image is negative that means the result varies on the interval [0,0.5]. For the trust result that interval is normalize to [0,1]
        else:
            trust[j] = round(float((p - 0.5) * (-2)),
                             4)
            trust_percs[j] = format(trust[j] * 100, '.2f')
            results.append('Negativo')
        j = j + 1

    return render_template('classify_multiple.html', filenames=img_name, results=results, preds=trust,
                           pred_percs=trust_percs, model=sel_model)
