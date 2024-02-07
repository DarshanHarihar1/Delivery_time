import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('randomreg.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Convert the data dictionary into a DataFrame
    df = pd.DataFrame(data, index=[0])
    
    print(df)

    new_data = scalar.transform(df)

    output = regmodel.predict(new_data)
    
    print(output[0])

    return jsonify({'prediction': output[0]})

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the form
    data = {
        'Delivery_person_Age': float(request.form['Delivery_person_Age']),
        'Delivery_person_Ratings': float(request.form['Delivery_person_Ratings']),
        'Vehicle_condition': float(request.form['Vehicle_condition']),
        'multiple_deliveries': float(request.form['multiple_deliveries']),
        'TimeOrder_Hour': float(request.form['TimeOrder_Hour']),
        'distance': float(request.form['distance']),
        'Type_of_order': request.form['Type_of_order'],
        'Type_of_vehicle': request.form['Type_of_vehicle'],
        'Festival': request.form['Festival'],
        'City': request.form['City'],
        'Delivery_city': request.form['Delivery_city'],
        'Road_traffic_density': request.form['Road_traffic_density'],
        'Weather_conditions': request.form['Weather_conditions']
    }

    # Convert data into DataFrame
    df = pd.DataFrame(data, index=[0])

    # Perform scaling
    new_data = scalar.transform(df)

    # Make prediction
    prediction = regmodel.predict(new_data)

    # Pass prediction to the template
    return render_template('home.html', prediction_text='Predicted Time: {} minutes'.format(prediction[0]))


if __name__=="__main__":
    app.run(debug=True)
   