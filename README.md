# PhoneWatch AI — Teen Phone Addiction Detection

A production-grade Flask web application with ML-powered teen phone addiction prediction.

## Project Structure
```
teen_addiction_app/
├── app.py                          # Flask backend
├── knn_model.pkl                   # Trained KNN model
├── processed_data.csv              # Processed dataset
├── teen_phone_addiction_dataset.csv # Raw dataset
├── requirements.txt
└── templates/
    ├── base.html                   # Base layout (nav, footer)
    ├── index.html                  # Homepage with stats
    ├── predict.html                # Prediction form
    ├── dashboard.html              # Analytics charts
    └── about.html                  # Project info
```

# Run the app
python app.py

Then open: http://localhost:5000

## Pages
- `/`          → Homepage with dataset statistics
- `/predict`   → AI prediction form (21 features → daily usage hours)
- `/dashboard` → Interactive charts and analytics
- `/about`     → Project & model details

## Model
- Algorithm: KNN Regressor (k=2)
- Features: 21 behavioral & psychological
- Target: Daily_Usage_Hours
- Risk Levels: Low / Moderate / High / Critical
