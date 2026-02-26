from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os
import sys
import threading

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

training_lock = threading.Lock()


@app.route("/")
def home():
    return render_template("index.html")




@app.route("/train", methods=["GET", "POST"])
def train_route():
    try:
        if request.method == "GET":
            return render_template("train.html")


        if training_lock.locked():
            lg.warning("Training already in progress attempt detected.")
            return " Training already in progress. Please wait."

        with training_lock:
            lg.info("========== TRAINING STARTED ==========")

            train_pipeline = TrainingPipeline()
            accuracy = train_pipeline.run_pipeline()

            lg.info("========== TRAINING COMPLETED ==========")

            return f"Training Completed Successfully. Accuracy: {accuracy}"

    except Exception as e:
        lg.error(f"Training failed: {str(e)}")
        raise CustomException(e, sys)



@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            return render_template("prediction.html")

        lg.info("Prediction request received.")

        prediction_pipeline = PredictionPipeline(request)
        prediction_file_detail = prediction_pipeline.run_pipeline()

        lg.info("Prediction completed. Sending file.")

        return send_file(
            prediction_file_detail.prediction_file_path,
            download_name=prediction_file_detail.prediction_file_name,
            as_attachment=True
        )

    except Exception as e:
        lg.error(f"Prediction failed: {str(e)}")
        raise CustomException(e, sys)



@app.errorhandler(Exception)
def handle_exception(e):
    lg.error(f"Unhandled Exception: {str(e)}")
    return "Something went wrong. Please check logs.", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host="0.0.0.0", port=port, debug=False)
