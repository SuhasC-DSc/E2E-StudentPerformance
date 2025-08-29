from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for Home Page
@app.route("/")
def home():
    intro = {
        "name": "Suhas Chandrashekaran",
        "title": "Aspiring Data Scientist & ML Enthusiast",
        "about": "I am passionate about building machine learning models, data-driven applications, and exploring AI. I also enjoy working on web development projects and experimenting with new technologies.",
        "skills": ["Python", "Machine Learning", "Flask", "SQL", "MongoDB", "Docker"],
        "contact": {
            "email": "suhas.chandrashekaran@gmail.com",
            "linkedin": "https://linkedin.com/in/suhas-chandrashekaran",
            "github": "https://github.com/SuhasC-DSc"
        }
    }
    return render_template("index.html", intro=intro)


# Route for Student Predictor Form
@app.route("/home")
def student_form():
    return render_template("home.html", results=None, error=None)


# Route for Prediction
@app.route("/predict", methods=["POST"])
def predict_datapoint():
    try:
        data = CustomData(
            gender=request.form.get("Gender"),
            race_ethnicity=request.form.get("RaceEthnicity"),
            parental_level_of_education=request.form.get("ParentalLevelOfEducation"),
            lunch=request.form.get("Lunch"),
            test_preparation_course=request.form.get("TestPreparationCourse"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )
    except (TypeError, ValueError):
        return render_template("home.html", results=None, error="⚠️ Please enter valid numeric values for scores.")

    # Convert input to DataFrame
    pred_df = data.get_data_as_dataframe()

    # Predict using your ML pipeline
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    return render_template("home.html", results=results[0], error=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)