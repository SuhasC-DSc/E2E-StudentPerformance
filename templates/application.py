from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for Home Page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            data = CustomData(
                gender=request.form.get('Gender'),
                race_ethnicity=request.form.get('RaceEthnicity'),
                parental_level_of_education=request.form.get('ParentalLevelOfEducation'),
                lunch=request.form.get('Lunch'),
                test_preparation_course=request.form.get('TestPreparationCourse'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
        except (TypeError, ValueError):
            return render_template('home.html', error="Please enter valid numeric values for scores.")

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

    return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
