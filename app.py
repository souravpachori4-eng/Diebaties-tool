from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import os

app = Flask(__name__)

# Put modified_dataset.csv in the same folder as this app.py,
# or change this path to your CSV file.
CSV_PATH = Path(__file__).with_name("modified_dataset.csv")

FEATURES = ["AGE", "Urea", "Cr", "HbA1c", "Chol", "TG",
            "HDL", "LDL", "VLDL", "BMI"]
TARGET = "CLASS_Encoded"

# This mapping follows the labels in your original Python code.
# Confirm that these encoded values match your dataset's meaning.
CLASS_NAMES = {
    0: "Non-diabetic",
    1: "Non-diabetic",
    2: "Prediabetic",
    3: "Diabetic",
    4: "Diabetic",
}

model = None
training_error = None


def train_model():
    global model, training_error
    try:
        if not CSV_PATH.exists():
            raise FileNotFoundError(
                f"CSV file not found: {CSV_PATH.name}. "
                "Place modified_dataset.csv beside app.py."
            )

        df = pd.read_csv(CSV_PATH)
        missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing_columns:
            raise ValueError("Missing CSV columns: " + ", ".join(missing_columns))

        data = df[FEATURES + [TARGET]].copy()
        for col in FEATURES + [TARGET]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=[TARGET])

        X = data[FEATURES]
        y = data[TARGET].astype(int)

        if len(data) < 5:
            raise ValueError("The dataset has too few usable rows to train.")
        if y.nunique() < 2:
            raise ValueError("The target column must contain at least two classes.")

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000))
        ])
        model.fit(X, y)
        training_error = None
    except Exception as exc:
        model = None
        training_error = str(exc)


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diabetes Risk Prediction</title>
<style>
:root { --navy:#102a43; --blue:#2563eb; --muted:#64748b; }
* { box-sizing:border-box; }
body {
  margin:0; font-family:Arial, sans-serif; color:#172554;
  background:linear-gradient(135deg,#eff6ff,#f8fafc 55%,#ecfeff);
  min-height:100vh;
}
header { padding:28px 18px 54px; color:white;
  background:linear-gradient(120deg,#123b70,#2563eb); }
.top { max-width:1000px; margin:auto; }
.tag { display:inline-block; padding:7px 12px; border:1px solid #ffffff66;
  border-radius:30px; font-size:12px; letter-spacing:.4px; }
h1 { margin:18px 0 8px; font-size:clamp(27px,5vw,42px); }
header p { margin:0; color:#dbeafe; line-height:1.5; }
main { max-width:1000px; margin:-28px auto 35px; padding:0 16px; }
.card { background:white; border:1px solid #e2e8f0; border-radius:20px;
  padding:22px; box-shadow:0 12px 35px #0f172a12; }
.section-title { margin:0 0 5px; font-size:20px; }
.sub { color:var(--muted); font-size:14px; margin:0 0 20px; line-height:1.5; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
label { display:block; font-size:13px; font-weight:700; margin-bottom:7px; color:#334155; }
input { width:100%; padding:13px 12px; border:1px solid #cbd5e1;
  border-radius:10px; font-size:16px; outline:none; background:#fff; }
input:focus { border-color:var(--blue); box-shadow:0 0 0 3px #2563eb20; }
button { width:100%; margin-top:22px; border:0; border-radius:12px;
  background:var(--blue); color:white; padding:15px; font-size:16px;
  font-weight:700; cursor:pointer; }
button:hover { background:#1d4ed8; }
.result { margin-top:20px; padding:18px; border-radius:14px; background:#eff6ff;
  border:1px solid #bfdbfe; }
.result h2 { margin:0 0 7px; font-size:21px; }
.result p { margin:5px 0; color:#334155; line-height:1.5; }
.error { margin-top:18px; padding:14px; border-radius:12px;
  background:#fff1f2; color:#9f1239; border:1px solid #fecdd3; }
.note { margin:18px 2px 0; color:#64748b; font-size:12px; line-height:1.55; }
footer { text-align:center; color:#64748b; padding:0 18px 28px; font-size:12px; }
@media(max-width:600px) { .grid { grid-template-columns:1fr; } .card {padding:18px;} }
</style>
</head>
<body>
<header><div class="top">
  <span class="tag">MACHINE LEARNING PROJECT</span>
  <h1>Diabetes Risk Prediction</h1>
  <p>Enter the health measurements below to view the model's predicted class.</p>
</div></header>
<main>
<div class="card">
  <h2 class="section-title">Patient information</h2>
  <p class="sub">Enter values using the same units and measurement conventions as your training dataset.</p>
  <form method="POST">
    <div class="grid">
      {% for key, label, hint in fields %}
      <div>
        <label for="{{key}}">{{label}}</label>
        <input id="{{key}}" name="{{key}}" type="number" step="any"
               placeholder="{{hint}}" value="{{values.get(key,'')}}" required>
      </div>
      {% endfor %}
    </div>
    <button type="submit">Predict diabetes class</button>
  </form>

  {% if result %}
  <div class="result">
    <h2>{{result}}</h2>
    <p>Model output class: <b>{{predicted_class}}</b></p>
    {% if confidence is not none %}
    <p>Model confidence for this class: <b>{{confidence}}%</b></p>
    {% endif %}
    <p>This is a machine-learning estimate, not a medical diagnosis.</p>
  </div>
  {% endif %}
  {% if error %}<div class="error">{{error}}</div>{% endif %}
</div>
<p class="note"><b>Important:</b> This is an educational portfolio project. It has not been clinically validated and must not be used to diagnose diabetes or guide treatment. A healthcare professional should interpret laboratory results.</p>
</main>
<footer>Diabetes Prediction • Student ML Project</footer>
</body>
</html>
"""

FIELDS = [
    ("AGE", "Age (years)", "e.g. 45"),
    ("Urea", "Urea", "Dataset unit"),
    ("Cr", "Creatinine (Cr)", "Dataset unit"),
    ("HbA1c", "HbA1c (%)", "e.g. 6.2"),
    ("Chol", "Total cholesterol", "Dataset unit"),
    ("TG", "Triglycerides (TG)", "Dataset unit"),
    ("HDL", "HDL cholesterol", "Dataset unit"),
    ("LDL", "LDL cholesterol", "Dataset unit"),
    ("VLDL", "VLDL cholesterol", "Dataset unit"),
    ("BMI", "BMI (kg/m²)", "e.g. 24.5"),
]


@app.route("/", methods=["GET", "POST"])
def home():
    values = {}
    result = None
    predicted_class = None
    confidence = None
    error = training_error

    if request.method == "POST":
        try:
            if model is None:
                raise RuntimeError(training_error or "The model could not be trained.")
            row = {}
            for key in FEATURES:
                raw = request.form.get(key, "").strip()
                if not raw:
                    raise ValueError(f"Please enter {key}.")
                value = float(raw)
                if value < 0:
                    raise ValueError(f"{key} cannot be negative.")
                row[key] = value
                values[key] = raw

            sample = pd.DataFrame([row], columns=FEATURES)
            prediction = int(model.predict(sample)[0])
            predicted_class = prediction
            result = CLASS_NAMES.get(prediction, f"Unknown class ({prediction})")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                classes = list(model.named_steps["classifier"].classes_)
                confidence = round(float(probabilities[classes.index(prediction)]) * 100, 1)
        except Exception as exc:
            error = str(exc)

    return render_template_string(PAGE, fields=FIELDS, values=values,
                                  result=result, predicted_class=predicted_class,
                                  confidence=confidence, error=error)


if __name__ == "__main__":
    train_model()
    print("Open http://127.0.0.1:5000 in your phone browser.")
    app.run(host="0.0.0.0", port=5000, debug=False)
