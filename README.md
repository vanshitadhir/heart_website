# Heart Disease Prediction Analysis (Flask + HTML/JS)

Full-stack project: dataset, preprocessing, EDA, 6 models (Linear Regression, Logistic Regression, KNN, SVM, Naive Bayes, Random Forest) and a live prediction form.

## Local Run
```bash
pip install -r requirements.txt
python backend/app.py
```
Then open http://127.0.0.1:5000 (models train at startup, takes a few seconds).

## Deploy on Render
1. Push this project to a GitHub repository.
2. In Render ([dashboard.render.com](https://dashboard.render.com)), click **New +** -> **Web Service**.
3. Select your repository.
4. Render will auto-detect settings, or enter:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend.app:app`
5. Click **Deploy Web Service**.

## Structure
- `backend/app.py` : preprocessing pipeline, model training/evaluation, REST API
- `backend/__init__.py` : package marker for gunicorn module loading
- `frontend/` : index.html, style.css, app.js, vendor/chart.umd.js (works offline)
- `data/heart_dataset.csv` : dataset
- `Procfile` & `render.yaml` : deployment configuration for Render
