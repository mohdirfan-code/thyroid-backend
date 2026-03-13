# Thyroid Detection CDSS (Clinical Decision Support System)

The **Thyroid CDSS** is a high-fidelity, state-of-the-art diagnostic system combining a purely native HTML5/JS Vanila front-end with an advanced Python backend. 

The application utilizes an ensemble of 5 machine learning models running via a decoupled REST API structure. It identifies 16 clinical variables to ascertain real-time probability gradients distinguishing between distinct thyroid dysfunctions with ultra-high accuracy and XAI (Explainable AI) clinical interpretability reporting.

## 🏗 System Architecture

The project has fundamentally two layers designed to be detached for maximum stability:

1. **Frontend**: (GitHub Pages)
   Vanilla HTML, CSS, JavaScript logic employing a clinical 'Glassmorphism' UI. It tracks patient intake payloads, calculates animations, and processes RESTful JSON outputs securely.

2. **Backend**: (Render)
   FastAPI Python environment that manages the data pipeline (Scaling) and AI parallel processing streams, emitting the final diagnostic interpretation arrays back via JSON.

## 🧠 Core Engineering: The 5-Paradigm Consensus
This system does not blindly trust a single neural process. It queries 5 highly-distinct paradigms sequentially.

- **Random Forest Core**: Anchor framework establishing baseline decision pathways.
- **XGBoost Module**: Boosted tree sequences resolving gradient variances.
- **TabNet AutoEncoder**: Sequential attention tracking dominant scalar dynamics natively inside tabular configurations.
- **KAN (Kolmogorov-Arnold Network)**: Advanced spline logic generating flexible feature bounds across non-linear inputs.
- **SAINT Transformer**: Self-attending tabular transformer processing categorical inter-dependencies.

The final result synthesizes these paradigms into a rigid logic tree generating the *"Primary Driver"* explanation and final prediction classification confidently.

---

## 🚀 Running Live (Render & GitHub Pages)

### 1. Render Deployment (Backend)
The `Procfile`, `requirements.txt`, and `main.py` are explicitly tailored to run out-of-the-box on Render's Linux containers.

- Hook a GitHub repo containing backend scripts exactly.
- Deploy a New **Web Service** on Render mapping that repository.
- Ensure the Render **Build Command** runs `pip install -r requirements.txt`.
- Set the **Start Command** mapping directly via Procfile natively: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- *Note: Torch uses `--index-url https://download.pytorch.org/whl/cpu` keeping dependencies small enough to run smoothly in standard Render environments.*

### 2. GitHub Pages Deployment (Frontend)
- The assets require absolute basic hosting, guaranteeing virtually limitless uptime.
- In `script.js` update `const API_URL` to point to the secure URL Render gives you (e.g. `https://thyroid-engine.onrender.com/predict`).
- Publish your frontend repo purely using the GitHub Pages standard deployment tools on branch `main`.

## 💻 Running Locally

To work on the codebase natively:

### Backend:
1. Active an environment (Python 3.10+ recommended).
2. Install via shell: `pip install -r requirements.txt`.
3. Launch the FastAPI server locally: `uvicorn main:app --reload`.
*(It will run securely on `http://127.0.0.1:8000`)*

### Frontend:
1. Open up `script.js` and change `const API_URL = 'http://127.0.0.1:8000/predict'`.
2. Double click the `index.html` file into any modern Chromium/WebKit browser natively mapping directly to the live environment.
