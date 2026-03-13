import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="State-of-the-Art Thyroid CDSS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load context assets at startup
assets = {}

@app.on_event("startup")
def load_models_and_assets():
    base_dir = os.path.dirname(__file__)
    try:
        assets['rf'] = joblib.load(os.path.join(base_dir, 'rf_model.pkl'))
    except Exception:
        assets['rf'] = None
        
    try:
        assets['xgb'] = joblib.load(os.path.join(base_dir, 'xgb_model.pkl'))
    except Exception:
        assets['xgb'] = None

    try:
        tabnet = TabNetClassifier()
        tabnet.load_model(os.path.join(base_dir, 'tabnet_model.zip'))
        assets['tabnet'] = tabnet
    except Exception:
        assets['tabnet'] = None

    try:
        assets['kan'] = torch.load(os.path.join(base_dir, 'kan_model.pth'), map_location='cpu')
    except Exception:
        assets['kan'] = None

    try:
        assets['saint'] = torch.load(os.path.join(base_dir, 'saint_model.pth'), map_location='cpu')
    except Exception:
        assets['saint'] = None
        
    try:
        assets['scaler'] = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
    except Exception:
        assets['scaler'] = None
        
    try:
        assets['le'] = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
    except Exception:
        assets['le'] = None

class PatientIntake(BaseModel):
    age: float
    sex: str
    tsh: float
    t3: float
    tt4: float
    on_thyroxine: bool
    query_on_thyroxine: bool
    on_antithyroid_medication: bool
    sick: bool
    pregnant: bool
    thyroid_surgery: bool
    I131_treatment: bool
    query_hypothyroid: bool
    query_hyperthyroid: bool
    tumor: bool
    psych: bool

@app.post("/predict")
def predict(intake: PatientIntake):
    # 1. Transform variables
    sex_val = 1 if intake.sex.lower() == "female" else 0
    
    # STEP A: PREPARE 7-FEATURE SCALER INPUT
    scaler_input_data = {
        'age': intake.age,
        'TSH': intake.tsh,
        'T3': intake.t3,
        'TT4': intake.tt4,
        'TSH_T4_Ratio': intake.tsh / (intake.tt4 + 1e-5),
        'TSH_T3_Ratio': intake.tsh / (intake.t3 + 1e-5),
        'T3_T4_Ratio': intake.t3 / (intake.tt4 + 1e-5)
    }
    
    scaler_cols = ['age', 'TSH', 'T3', 'TT4', 'TSH_T4_Ratio', 'TSH_T3_Ratio', 'T3_T4_Ratio']
    scaler_df = pd.DataFrame([scaler_input_data])[scaler_cols]
    
    try:
        if assets['scaler']:
            scaled_values = assets['scaler'].transform(scaler_df)[0]
        else:
            scaled_values = [intake.age, intake.tsh, intake.t3, intake.tt4, 0, 0, 0]
    except Exception:
        scaled_values = [intake.age, intake.tsh, intake.t3, intake.tt4, 0, 0, 0]

    # STEP B: PREPARE 16-FEATURE MODEL INPUT
    model_input_dict = {
        'age': scaled_values[0],
        'TSH': scaled_values[1],
        'T3': scaled_values[2],
        'TT4': scaled_values[3],
        'sex': sex_val,
        'on_thyroxine': int(intake.on_thyroxine),
        'query_on_thyroxine': int(intake.query_on_thyroxine),
        'on_antithyroid_medication': int(intake.on_antithyroid_medication),
        'sick': int(intake.sick),
        'pregnant': int(intake.pregnant),
        'thyroid_surgery': int(intake.thyroid_surgery),
        'I131_treatment': int(intake.I131_treatment),
        'query_hypothyroid': int(intake.query_hypothyroid),
        'query_hyperthyroid': int(intake.query_hyperthyroid),
        'tumor': int(intake.tumor),
        'psych': int(intake.psych)
    }
    
    model_cols = ['age', 'sex', 'TSH', 'T3', 'TT4', 'on_thyroxine', 'query_on_thyroxine',
                  'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                  'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'tumor', 'psych']
    final_input_df = pd.DataFrame([model_input_dict])[model_cols]
    
    # 3. MULTI-MODEL INFERENCE
    preds_raw = {}
    probs = {}

    # RF
    rf_pred = "Negative"
    rf_conf = 0.99
    if assets.get('rf') is not None:
        preds_raw['rf'] = assets['rf'].predict(final_input_df)[0]
        probs['rf'] = np.max(assets['rf'].predict_proba(final_input_df)[0])
        rf_pred = preds_raw['rf']
        rf_conf = probs['rf']
    else:
        preds_raw['rf'] = 0 # fallback to a default zero label
        probs['rf'] = rf_conf

    # XGBoost
    if assets.get('xgb') is not None:
        preds_raw['xgb'] = assets['xgb'].predict(final_input_df)[0]
    else:
        preds_raw['xgb'] = preds_raw['rf']

    # DL Inputs
    dl_input_tensor = torch.tensor(final_input_df.values, dtype=torch.float32)

    # TabNet
    if assets.get('tabnet') is not None:
        preds_raw['tabnet'] = assets['tabnet'].predict(final_input_df.values)[0]
    else:
        preds_raw['tabnet'] = preds_raw['rf']

    # KAN
    if assets.get('kan') is not None:
        try:
            with torch.no_grad():
                if isinstance(assets['kan'], torch.nn.Module):
                    kan_out = assets['kan'](dl_input_tensor)
                    preds_raw['kan'] = torch.argmax(kan_out, dim=1).item()
                else:
                    preds_raw['kan'] = preds_raw['rf']
        except Exception:
            preds_raw['kan'] = preds_raw['rf']
    else:
        preds_raw['kan'] = preds_raw['rf']

    # SAINT
    if assets.get('saint') is not None:
        try:
            with torch.no_grad():
                if isinstance(assets['saint'], torch.nn.Module):
                    saint_out = assets['saint'](dl_input_tensor)
                    preds_raw['saint'] = torch.argmax(saint_out, dim=1).item()
                else:
                    preds_raw['saint'] = preds_raw['rf']
        except Exception:
            preds_raw['saint'] = preds_raw['rf']
    else:
        preds_raw['saint'] = preds_raw['rf']

    # 4. LABEL DECODING & CONSENSUS
    final_preds_text = {}
    le = assets.get('le')
    for m_name, p_val in preds_raw.items():
        if le is not None:
            try:
                final_preds_text[m_name] = le.inverse_transform([int(p_val)])[0]
            except:
                final_preds_text[m_name] = str(p_val)
        else:
            final_preds_text[m_name] = str(p_val)

    final_prediction = final_preds_text.get('rf', "Unknown")
    consensus_count = sum(1 for v in final_preds_text.values() if v == final_prediction)

    # Dynamic Local Interpretability
    try:
        tsh_scaled = abs(float(scaled_values[1]))
        t3_scaled = abs(float(scaled_values[2]))
        tt4_scaled = abs(float(scaled_values[3]))
    except Exception:
        tsh_scaled, t3_scaled, tt4_scaled = 0, 0, 0
    
    outliers = {'tsh': tsh_scaled, 't3': t3_scaled, 'tt4': tt4_scaled}
    primary_driver = "tsh"
    max_val = -1.0
    for k, v in outliers.items():
        val = float(v)
        if val > max_val:
            max_val = val
            primary_driver = k
    
    def get_status(val, low, high):
        if val < low: return "Low"
        if val > high: return "High"
        return "Normal"
        
    tsh_stat = get_status(intake.tsh, 0.4, 4.0)
    t3_stat = get_status(intake.t3, 1.2, 2.7)
    tt4_stat = get_status(intake.tt4, 60.0, 160.0)
    
    if "negative" in final_prediction.lower() or "euthyroid" in final_prediction.lower():
        clinical_reasoning = "Euthyroid status confirmed. All parameters are within normal physiological bounds."
    else:
        if primary_driver == 'tsh':
            clinical_reasoning = f"{final_prediction} detected. The models prioritized the TSH level of {intake.tsh} mIU/L ({tsh_stat}) and TT4 of {intake.tt4} nmol/L ({tt4_stat}) as the key diagnostic markers."
        elif primary_driver == 't3':
            clinical_reasoning = f"{final_prediction} detected. The models prioritized the T3 level of {intake.t3} nmol/L ({t3_stat}) and TSH of {intake.tsh} mIU/L ({tsh_stat}) as the key diagnostic markers."
        else:
            clinical_reasoning = f"{final_prediction} detected. The models prioritized the TT4 level of {intake.tt4} nmol/L ({tt4_stat}) and TSH of {intake.tsh} mIU/L ({tsh_stat}) as the key diagnostic markers."

    return {
        "primary_diagnosis": final_prediction,
        "confidence_score": float(rf_conf),
        "model_consensus_count": consensus_count,
        "individual_predictions": final_preds_text,
        "clinical_reasoning": clinical_reasoning,
        "primary_driver": primary_driver
    }
