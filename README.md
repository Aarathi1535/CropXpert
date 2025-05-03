# 🌾 CropXpert – Your AI Agriculture Partner

![Crop Recommendation](/static/crop-recommend.jpg)

**CropXpert** is a comprehensive AI-powered web application designed to assist farmers and agricultural professionals in making informed decisions about crop and fertilizer selection. Built using an end-to-end stack of machine learning, web technologies, and database systems, CropXpert aims to revolutionize smart farming practices by delivering real-time, data-driven recommendations.

---

## 🚀 Project Overview

Choosing the right crop or fertilizer is often a challenge due to various environmental and soil-based factors. CropXpert addresses this by leveraging machine learning to predict:

- **Best Crop to Cultivate** based on soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, and rainfall.
- **Optimal Fertilizer Recommendation** using soil content, crop type, and environmental parameters.

With a user-friendly interface and seamless backend integration, the app serves as an intelligent assistant to help farmers increase productivity and efficiency.

---

## 🛠️ Tech Stack & Tools Used

- **Machine Learning**: Decision Tree Classifier
- **Data Preprocessing**: Label Encoder (for string-valued dataset columns)
- **Programming Language**: Python
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Database**: PostgreSQL (via SQLAlchemy ORM)
- **Visualization Tools**: Seaborn, Matplotlib, Confusion Matrix
- **Libraries**:
  - `scikit-learn`
  - `pandas`
  - `flask`
  - `SQLAlchemy`
  - `seaborn`
  - `matplotlib`

---

## 🧭 Workflow Diagrams

### 🔧 Technical Architecture  
![Technical](/WorkflowDiagram.png)

### 🐛 Pest Detection Flow  
![Pest](/PestDetectionFlowchart.png)

---

## 🌐 Deployment Details

The Flask app is deployed on **Render**.

### 🔧 Build & Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app using Gunicorn (Render Deployment)
gunicorn -b :$PORT app:app

---

## 📊 Crop Recommendation – Analysis Report

![Report](/static/Report-image.png)

---

## 🔗 Live Demo Links

- 🌿 **Try the App Now**: [CropXpert Live on Render](https://crop-recommendation-system-app.onrender.com)
- 💡 **Full Feature Preview**: [Enhanced Version of CropXpert](https://cropxpert.lovable.app/)
- 🎥 **Video Demonstration**: [Watch on YouTube](https://youtu.be/c43HSkkh4GY)

---

## 💡 What Makes CropXpert Unique?

- ✅ Intuitive, farmer-friendly UI  
- ✅ Accurate crop and fertilizer predictions using real-time data  
- ✅ Pest detection module through image input and machine learning models  
- ✅ Seamless integration of ML models, web technologies, and database systems  
- ✅ Fully deployable and scalable for production environments  

---

## 📬 Contact

For queries, collaborations, or feedback, feel free to reach out:

- 📧 **Email**: aarathisree.1535@gmail.com  
- 📱 **Phone**: +91 9381481266
