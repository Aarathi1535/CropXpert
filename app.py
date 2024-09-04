from flask import Flask, render_template, request, redirect, url_for, session, flash
#from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from keras.models import load_model
from matplotlib import pyplot as plt
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'Aarathi@1535'

# Database setup (e.g., SQLite)
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, 
               name TEXT, email TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def execute_query(query, args=()):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(query, args)
    conn.commit()
    conn.close()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')

        query = "INSERT INTO users (name, email, password) VALUES (?, ?, ?)"
        try:
            execute_query(query, (name, email, password))
            flash('You have successfully signed up! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('User with this email already exists.', 'danger')
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            print("Login successful, redirecting to home")
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check your email and password', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have successfully logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user_name=session.get('user_name'))


@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/cropdetails', methods=["POST"])
def cropdetails():
    df5 = pd.read_csv("crop_info.csv")
    cropname = str(request.form['cropname']).lower()
    results = ''
    if cropname in df5['Crop Name'].values:
        results = df5[df5['Crop Name'] == cropname]['Info'].iloc[0]
    return render_template('cropdetails.html', result=results)

# Crop Recommendation Section
df = pd.read_csv("Crop_recommendation.csv")
df1 = df.drop(['Unnamed: 8', 'Unnamed: 9'], axis=1)
x = df1.drop(['label'], axis=1)
y = df1['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

@app.route('/result', methods=["POST"])
def result():
    nitrogen = int(request.form['nitrogen'])
    phosphorus = int(request.form['phosphorus'])
    potassium = int(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    prediction = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    result = (
        f"As per your given inputs:<br> "
        f"Nitrogen Content = {nitrogen}<br> "
        f"Phosphorus Content = {phosphorus}<br> "
        f"Potassium Content = {potassium}<br> "
        f"Temperature = {temperature}<br> "
        f"Humidity = {humidity}<br> "
        f"pH value = {ph}<br> "
        f"Rainfall in mm = {rainfall}<br>"
        f"The best crop that suits your inputs is: <strong>{prediction[0]}</strong>"
    )
    return render_template('result.html', result=result)

# Fertilizer Recommendation Section
df2 = pd.read_csv("Fertilizer Prediction.csv")
le = LabelEncoder()
df2['New_Soil_type'] = le.fit_transform(df2['Soil Type'])
df2['New_Crop_type'] = le.fit_transform(df2['Crop Type'])
df3 = df2
df3 = df3.drop(['Soil Type','Crop Type'],axis=1)
a = df3.drop(['Fertilizer Name'],axis=1)
b = df3['Fertilizer Name']
a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.25)
model2 = RandomForestClassifier()
model2.fit(a_train,b_train)

@app.route('/result2', methods=["POST"])
def result2():
    temp = int(request.form['temp'])
    humid = int(request.form['humid'])
    moisture = int(request.form['moisture'])
    nitro = int(request.form['nitro'])
    potash = int(request.form['potash'])
    phospho = int(request.form['phospho'])
    soil = int(request.form['soil'])
    crop = int(request.form['crop'])
    prediction = model2.predict([[temp, humid, moisture, nitro, potash, phospho, soil, crop]])
    result = (
        f"As per your given inputs:<br> "
        f"Temperature = {temp}<br> "
        f"Humidity = {humid}<br> "
        f"Moisture = {moisture}<br> "
        f"Nitogren Content = {nitro}<br> "
        f"Potassium Content = {potash}<br> "
        f"Phosphorus Content = {phospho}<br> "
        f"Soil Type = {soil}<br>"
        f"Crop Type = {crop}<br>"
        f"The best Fertilizer that suits your inputs is: <strong>{prediction[0]}</strong>"
    )
    return render_template('result2.html', result=result)
'''
# Image Upload and Prediction Section
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the file to a temporary location
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Make a prediction
        predicted_class = predict_pest(file_path)
        predicted_label = imagenet_labels[predicted_class - 1]

        # Return the result to the pest_detection.html template
        return render_template('pest_detection.html', label=predicted_label)

    return render_template('index.html')

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Load the trained model
model = load_model('pest_detection_model.h5')

# Define a mapping from class indices to pest names
class_index_to_name = {
    0: "Aphids",
    1: "Armyworm",
    2: "Beetle",
    3: "Bollworm",
    4: "Grasshopper",
    5: "Mites",
    6: "Mosquito",
    7: "Sawfly",
    8: "Stem borer"
}

# Function to get the image path of the pest (update as needed)
def get_pest_image_path(pest_name):
    base_dir = r"C:\\Users\\AARATHISREE\\Desktop\\Jupyter Notebook Projects\\Pest_datasets\\pest\\train"
    pest_dir = os.path.join(base_dir, pest_name)
    return os.path.join(pest_dir, f'{pest_name}_jpg_0.jpg')

# Function to predict the pest
def predict_pest(image_path):
    img_size = 128
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    pest_name = class_index_to_name.get(predicted_class, "Unknown")

    pest_image_path = get_pest_image_path(pest_name)
    return pest_name, pest_image_path

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Predict the pest
        pest_name, pest_image_path = predict_pest(file_path)

        return render_template('pest_detection.html', pest_name=pest_name, pest_image_path=pest_image_path, uploaded_image=file.filename)
    
'''
# Ensure the uploads directory exists
if __name__ == '__main__':
    init_db()
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
