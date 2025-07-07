from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils.predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    
    result = None
    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename != "":
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filepath)
                result = predict_image(filepath)
                result["ImagePath"] = filepath

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
