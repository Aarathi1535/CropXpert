from flask import Flask,render_template,request # type: ignore
from sklearn import tree
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\My Desktop\Crop_recommendation.csv")
df1 = df.drop(df[['Unnamed: 8','Unnamed: 9']],axis=1)
x = df1.drop(['label'],axis=1)
y = df1['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/result',methods=["POST"])
def result():
    nitrogen = int(request.form['nitrogen'])
    phosphorus = int(request.form['phosphorus'])
    potassium = int(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    prediction = model.predict([[nitrogen,phosphorus,potassium,temperature,humidity,ph,rainfall]])
    result = (
        f"As per your given inputs:<br> "
        f"Nitrogen Content = {nitrogen}<br> "
        f"Phosphorus Content = {phosphorus}<br> "
        f"Potassium Content = {potassium}<br> "
        f"Temperature = {temperature}<br> "
        f"Humidity = {humidity}<br> "
        f"pH value = {ph}<br> "
        f"Rainfall in mm = {rainfall}<br>"
        f"The best crop that suits your inputs is: <strong>{prediction[0]}"
    )
    return render_template('result.html',result = result)

if __name__ == '__main__':
    app.run(debug=True)