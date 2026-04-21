from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load model and data
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

df_raw = pd.read_csv("teen_phone_addiction_dataset.csv")
df_processed = pd.read_csv("processed_data.csv")

# Encode mappings (same as training)
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
GRADE_MAP = {"7th": 3, "8th": 4, "9th": 5, "10th": 0, "11th": 1, "12th": 2}

def get_dashboard_stats():
    stats = {
        "total_students": len(df_raw),
        "avg_daily_usage": round(df_raw["Daily_Usage_Hours"].mean(), 1),
        "avg_addiction": round(df_raw["Addiction_Level"].mean(), 1),
        "high_addiction_pct": round((df_raw["Addiction_Level"] >= 7).sum() / len(df_raw) * 100, 1),
        "avg_sleep": round(df_raw["Sleep_Hours"].mean(), 1),
        "avg_academic": round(df_raw["Academic_Performance"].mean(), 1),
    }
    return stats

def get_chart_data():
    # Addiction by grade
    grade_group = df_raw.groupby("School_Grade")["Addiction_Level"].mean().reset_index()
    grade_order = ["7th", "8th", "9th", "10th", "11th", "12th"]
    grade_group["School_Grade"] = pd.Categorical(grade_group["School_Grade"], categories=grade_order, ordered=True)
    grade_group = grade_group.sort_values("School_Grade")

    # Addiction by gender
    gender_group = df_raw.groupby("Gender")["Addiction_Level"].mean().reset_index()

    # Usage distribution
    bins = [0, 2, 4, 6, 8, 24]
    labels = ["0-2h", "2-4h", "4-6h", "6-8h", "8h+"]
    df_raw["usage_bin"] = pd.cut(df_raw["Daily_Usage_Hours"], bins=bins, labels=labels)
    usage_dist = df_raw["usage_bin"].value_counts().reindex(labels).reset_index()

    # Anxiety vs Addiction scatter (sample 200)
    sample = df_raw.sample(min(200, len(df_raw)), random_state=42)

    return {
        "grade_labels": grade_group["School_Grade"].tolist(),
        "grade_addiction": grade_group["Addiction_Level"].round(2).tolist(),
        "gender_labels": gender_group["Gender"].tolist(),
        "gender_addiction": gender_group["Addiction_Level"].round(2).tolist(),
        "usage_labels": labels,
        "usage_counts": usage_dist["count"].tolist(),
        "scatter_anxiety": sample["Anxiety_Level"].tolist(),
        "scatter_addiction": sample["Addiction_Level"].tolist(),
        "scatter_age": sample["Age"].tolist(),
    }

@app.route("/")
def index():
    stats = get_dashboard_stats()
    return render_template("index.html", stats=stats)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            data = request.form
            features = [
                int(data.get("id", 999)),
                int(data["age"]),
                GENDER_MAP.get(data["gender"], 0),
                GRADE_MAP.get(data["school_grade"], 5),
                float(data["sleep_hours"]),
                int(data["academic_performance"]),
                int(data["social_interactions"]),
                float(data["exercise_hours"]),
                int(data["anxiety_level"]),
                int(data["depression_level"]),
                int(data["self_esteem"]),
                int(data["parental_control"]),
                float(data["screen_time_before_bed"]),
                int(data["phone_checks_per_day"]),
                int(data["apps_used_daily"]),
                float(data["time_on_social_media"]),
                float(data["time_on_gaming"]),
                float(data["time_on_education"]),
                int(data["family_communication"]),
                float(data["weekend_usage_hours"]),
                float(data["addiction_level"]),
            ]
            prediction = model.predict([features])[0]
            prediction = round(float(prediction), 2)

            # Risk classification
            if prediction <= 3:
                risk = "Low"
                risk_color = "low"
                advice = "Great! This teen shows healthy phone usage habits. Encourage outdoor activities and maintain current boundaries."
            elif prediction <= 5:
                risk = "Moderate"
                risk_color = "moderate"
                advice = "This teen shows moderate phone usage. Consider setting screen time limits and encouraging more offline activities."
            elif prediction <= 7:
                risk = "High"
                risk_color = "high"
                advice = "Warning: This teen shows high phone usage. Parental intervention is recommended. Introduce tech-free zones."
            else:
                risk = "Critical"
                risk_color = "critical"
                advice = "Alert: Critical addiction level detected. Immediate intervention required. Consider consulting a digital wellness specialist."

            result = {
                "prediction": prediction,
                "risk": risk,
                "risk_color": risk_color,
                "advice": advice,
            }
        except Exception as e:
            result = {"error": str(e)}
    return render_template("predict.html", result=result)

@app.route("/dashboard")
def dashboard():
    chart_data = get_chart_data()
    return render_template("dashboard.html", chart_data=chart_data)

@app.route("/api/chart-data")
def api_chart_data():
    return jsonify(get_chart_data())

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
