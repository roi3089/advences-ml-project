# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import joblib
import pandas as pd
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Helper function to normalize manufacturer names
def normalize_manufacturer(name):
    name = name.lower()
    if 'mazda' in name or 'מזדה' in name or 'מאזדה' in name:
        return 'Mazda'
    if 'hyundai' in name or 'יונדאי' in name:
        return 'Hyundai'
    if 'toyota' in name or 'טויוטה' in name:
        return 'Toyota'
    if 'ford' in name or 'פורד' in name:
        return 'Ford'
    if 'peugeot' in name or 'פיגו' in name or "פיג'ו" in name:
        return 'Peugeot'
    if 'kia' in name or 'קאיה' in name or 'קיה' in name:
        return 'Kia'
    return name.capitalize()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {
        'Year': [request.form.get('year', type=int)],
        'manufactor': [normalize_manufacturer(request.form.get('manufacturer', default='Unknown'))],
        'model': [request.form.get('model', default='Unknown')],
        'Hand': [request.form.get('hand', type=int, default=1)],
        'Gear': [request.form.get('gear', default='Manual')],
        'capacity_Engine': [request.form.get('capacity_engine', type=float, default=1.0)],
        'Engine_type': [request.form.get('engine_type', default='Petrol')],
        'Prev_ownership': [request.form.get('prev_ownership', default='1')],
        'Curr_ownership': [request.form.get('curr_ownership', default='1')],
        'Area': [request.form.get('area', default='Unknown')],
        'City': [request.form.get('city', default='Unknown')],
        'Km': [request.form.get('km', type=float, default=0.0)],
        'Test': [request.form.get('test', default='2022-01-01')],
        'Pic_num': [request.form.get('pic_num', type=int, default=1)],
        'Color': [request.form.get('color', default='Unknown')]
    }

    # Create DataFrame from the form data
    df = pd.DataFrame(data)

    # Process the data using the prepare_data function
    processed_data = prepare_data(df)

    # Predict the price using the trained model
    predicted_price = model.predict(processed_data)[0]

    # Ensure that the predicted price is not negative
    if predicted_price < 0:
        predicted_price = "Error - Invalid Prediction"

    # Render the home template with the predicted price
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
