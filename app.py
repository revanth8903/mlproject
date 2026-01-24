
from flask import Flask,render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application

## Route for a Home Page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        ## I'm reading "Gender" by using Request.form.get, When we do the Post, this request will have the entire information
        data=CustomData(                                              ## We will just try to read all the variable values  
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])  
    
     ## [0] This will be in the list format, We need to read this value in "home.html"

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)  ## This host address is going to map up with "127.0.1" with debug=True

