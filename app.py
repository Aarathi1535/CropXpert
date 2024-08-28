from flask import Flask, render_template, request, redirect, url_for
#from flask_sqlalchemy import SQLAlchemy
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
import os

warnings.filterwarnings('ignore')

# Load the ImageNet labels from the JSON file
with open("imagenet-simple-labels.json", 'r') as file:
    imagenet_labels = json.load(file)

# Load the pre-trained ResNet model
model3 = models.resnet18(pretrained=True)
model3.eval()  

# Define the transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  
    return image

def predict_pest(image_path):
    image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model3(image)

    # Get the predicted class
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

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

# Ensure the uploads directory exists
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
