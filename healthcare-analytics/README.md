# Healthcare Predictive Analytics — Disease Detection

A browser-based predictive analytics tool for disease risk detection (Diabetes & Heart Disease)
built with vanilla HTML, CSS, and JavaScript. Uses real UCI/Kaggle dataset features and
weighted classification models for risk prediction.

---

## Features

- **Diabetes Risk Prediction** — Pima Indians Diabetes Dataset (UCI), Random Forest model
- **Heart Disease Risk Prediction** — Cleveland Heart Disease Dataset (UCI), Gradient Boosting model
- **Model Insights** — Feature importance charts, AUC-ROC, accuracy, confusion matrix metrics
- **Ethics & Privacy Panel** — Data handling, bias mitigation, HIPAA compliance notes
- Interactive sliders and dropdowns for patient health indicators
- Animated risk meter with feature contribution breakdown
- Fully client-side — no data transmitted or stored

---

## Project Structure

```
healthcare-analytics/
├── index.html          ← Main HTML entry point
├── css/
│   └── style.css       ← All styles (sidebar, cards, sliders, charts)
├── js/
│   └── app.js          ← Prediction logic + Chart.js rendering
└── README.md           ← This file
```

---

## How to Run in VS Code

### Option 1 — Live Server Extension (Recommended)

1. **Open VS Code**

2. **Open the project folder**
   - Go to `File → Open Folder`
   - Select the `healthcare-analytics` folder
   - Click **Open**

3. **Install the Live Server extension**
   - Click the Extensions icon in the left sidebar (or press `Ctrl+Shift+X`)
   - Search for **"Live Server"** by Ritwick Dey
   - Click **Install**

4. **Launch the app**
   - In the Explorer panel, right-click `index.html`
   - Select **"Open with Live Server"**
   - Your browser will open automatically at `http://127.0.0.1:5500`

5. **Any changes you save** in VS Code will auto-refresh the browser.

---

### Option 2 — VS Code Built-in Simple Browser

1. Open the project folder in VS Code (`File → Open Folder`)

2. Open the Terminal in VS Code:
   - Go to `Terminal → New Terminal`
   - Or press `` Ctrl+` ``

3. Start a local server using Python (usually pre-installed):
   ```bash
   # Python 3
   python -m http.server 8080

   # Python 2 (older systems)
   python -m SimpleHTTPServer 8080
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

---

### Option 3 — Node.js http-server

1. Make sure Node.js is installed: https://nodejs.org

2. Open Terminal in VS Code (`` Ctrl+` ``)

3. Install `http-server` globally (one-time):
   ```bash
   npm install -g http-server
   ```

4. Run from the project folder:
   ```bash
   http-server . -p 8080
   ```

5. Open `http://localhost:8080` in your browser.

---

## Dataset References

| Dataset | Source | Samples | Features |
|---|---|---|---|
| Pima Indians Diabetes | UCI ML Repository | 768 | 8 |
| Cleveland Heart Disease | UCI ML Repository | 303 | 13 |

- Diabetes: https://archive.ics.uci.edu/ml/datasets/diabetes
- Heart Disease: https://archive.ics.uci.edu/ml/datasets/heart+disease

---

## Technology Stack

| Tool | Purpose |
|---|---|
| HTML5 / CSS3 | Structure & styling |
| Vanilla JavaScript | Prediction logic & interactivity |
| Chart.js 4.4 | Feature importance charts |
| Tabler Icons | UI icon set |
| Python / scikit-learn | (Reference) model training |

---

## Ethical Notes

- All predictions are **local and client-side** — no data is stored or transmitted
- Models are approximations for **educational/research purposes only**
- The Pima dataset has known demographic limitations (female, Pima Indian heritage only)
- **This is NOT medical advice.** Always consult a licensed healthcare professional.

---

## License

MIT License — free to use for educational and research purposes.
