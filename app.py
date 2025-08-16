from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import math

app = Flask(__name__)

# ---------- Load trained model ----------
MODEL_PATH = os.path.join("model", "eia_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- Features used during training (order matters) ----------
MODEL_FEATURES = [
    "Year of reporting",
    "Log_Product_Weight",
    "PCF_Change_Category_Encoded",
    "Total_CO2e_Fraction",
    "Product_Detail_Word_Count",
    "Change_Reason_Word_Count",
]

# ---------- Helpers ----------
def safe_float(v, default=0.0):
    try:
        if v is None or str(v).strip() == "":
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)

def safe_int(v, default=0):
    try:
        if v is None or str(v).strip() == "":
            return int(default)
        return int(float(str(v).strip()))
    except Exception:
        return int(default)

def encode_change_from_relative(relative_change_decimal):
    """
    Uses same logic as training:
      >  +5%  -> Increase (1)
      <  -5%  -> Decrease (2)
      else    -> Stable  (0)
    NOTE: relative_change_decimal is a decimal (e.g., 0.12 for 12%)
    """
    if relative_change_decimal > 0.05:
        return 1  
    elif relative_change_decimal < -0.05:
        return 2  
    return 0      

def build_feature_row(form_or_json):
    """
    Builds a single-row DataFrame with exactly MODEL_FEATURES in correct order
    from form (request.form) or JSON (request.json).
    """

    # 1) Year of reporting
    year = safe_int(form_or_json.get("year_of_reporting"), 0)

    # 2) Product weight (kg) -> Log_Product_Weight = log1p(weight)
    weight_kg = safe_float(form_or_json.get("product_weight_kg"), 0.0)
    log_wt = math.log1p(weight_kg) if weight_kg >= 0 else 0.0

    # 3) Relative change (%) -> decimal -> encode category
    # e.g., user enters "12" → 0.12
    rel_change_pct = safe_float(form_or_json.get("relative_change_pct"), 0.0)
    rel_change_dec = rel_change_pct / 100.0
    pcf_change_cat_encoded = encode_change_from_relative(rel_change_dec)

    # 4) Stage fractions (%) — user can enter any/all, we sum them and convert to decimal total
    up_pct   = safe_float(form_or_json.get("upstream_pct"),   0.0)
    ops_pct  = safe_float(form_or_json.get("operations_pct"), 0.0)
    down_pct = safe_float(form_or_json.get("downstream_pct"), 0.0)
    trans_pct= safe_float(form_or_json.get("transport_pct"),  0.0)
    eol_pct  = safe_float(form_or_json.get("eol_pct"),        0.0)
    total_fraction = (up_pct + ops_pct + down_pct + trans_pct + eol_pct) / 100.0

    # 5) Word counts from texts
    product_detail = (form_or_json.get("product_detail") or "").strip()
    reason_text    = (form_or_json.get("change_reason") or "").strip()
    wc_detail = len(product_detail.split()) if product_detail else 0
    wc_reason = len(reason_text.split()) if reason_text else 0

    row = {
        "Year of reporting": year,
        "Log_Product_Weight": log_wt,
        "PCF_Change_Category_Encoded": pcf_change_cat_encoded,
        "Total_CO2e_Fraction": total_fraction,
        "Product_Detail_Word_Count": wc_detail,
        "Change_Reason_Word_Count": wc_reason,
    }

    X = pd.DataFrame([row])
    X = X.reindex(columns=MODEL_FEATURES, fill_value=0)
    return X

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    latest_inputs = None

    if request.method == "POST":
        try:
            X_user = build_feature_row(request.form)
            pred = model.predict(X_user)[0]
            prediction_value = round(float(pred), 6)
            prediction = f"Predicted Carbon Footprint is {prediction_value} kg CO₂e"
            latest_inputs = {k: v for k, v in request.form.items()}

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           model_features=MODEL_FEATURES,
                           latest_inputs=latest_inputs)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        X_user = build_feature_row(data)
        pred = model.predict(X_user)[0]
        return jsonify({
            "ok": True,
            "prediction": float(round(pred, 6)),
            "features_used": MODEL_FEATURES
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
