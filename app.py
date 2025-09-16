from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

model = joblib.load('best_obesity_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for col in ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']:
            input_data[col] = [request.form[col]]

        df_input = pd.DataFrame(input_data)

        for col, le in label_encoders.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col])

        X_scaled = scaler.transform(df_input)

        prediction = model.predict(X_scaled)
        predicted_class = prediction[0]

        for col, le in label_encoders.items():
            if col == 'NObeyesdad':
                predicted_class = le.inverse_transform([predicted_class])[0]

        return render_template('index.html', prediction_text=f'Predicted Obesity Class: {predicted_class}')
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
