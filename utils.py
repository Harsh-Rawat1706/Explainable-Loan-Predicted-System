import pickle
import pandas as pd
import shap

# ================= LOAD MODEL =================
with open("model/finalize_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# SHAP Explainer (load once)
explainer = shap.TreeExplainer(model)


# ================= INPUT PROCESSING =================
def process_input(form_data):
    data = {
        "no_of_dependents": int(form_data["dependents"]),
        "education": int(form_data["education"]),
        "self_employed": int(form_data["self_employed"]),
        "income_annum": float(form_data["income"]),
        "loan_amount": float(form_data["loan_amount"]),
        "loan_term": float(form_data["loan_term"]),
        "cibil_score": float(form_data["cibil"]),
        "residential_assets_value": float(form_data["res_assets"]),
        "commercial_assets_value": float(form_data["com_assets"]),
        "luxury_assets_value": float(form_data["lux_assets"]),
        "bank_asset_value": float(form_data["bank_assets"])
    }

    df = pd.DataFrame([data])

    # ensure correct column order
    df = df[feature_names]

    return df


# ================= MAIN FUNCTION =================
def predict_and_explain(df):

    # -------- PREDICTION --------
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # -------- SHAP --------
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values.flatten()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "value": df.iloc[0].values
    })

    # -------- FIND NEGATIVE FEATURES --------
    negative = (
        shap_df[shap_df["shap_value"] < 0]
        .sort_values("shap_value")
        .head(3)
    )

    suggestions = []

    # -------- HELPER FUNCTION (WHAT-IF SIMULATION) --------
    def simulate_feature_change(feature, start_value, step, direction, target_prob=0.6, max_iter=50):
        temp_df = df.copy()

        for i in range(max_iter):
            if direction == "increase":
                new_value = start_value + step * i
            else:
                new_value = start_value - step * i

            # safety limits
            if feature == "cibil_score":
                new_value = min(new_value, 900)
            if feature == "loan_amount":
                new_value = max(new_value, 10000)
            if feature == "loan_term":
                new_value = max(new_value, 1)

            temp_df[feature] = new_value

            new_prob = model.predict_proba(temp_df)[0][1]

            if new_prob >= target_prob:
                return new_value, new_prob

        return None, prob

    # -------- DYNAMIC SUGGESTIONS --------
    for feature in negative["feature"]:

        # safety check
        if feature not in df.columns:
            continue

        current_val = df[feature].values[0]

        # 🔹 CIBIL SCORE
        if feature == "cibil_score":
            new_val, new_prob = simulate_feature_change(
                feature, current_val, step=10, direction="increase"
            )
            if new_val:
                suggestions.append(
                    f"Increase CIBIL score from {int(current_val)} → {int(new_val)} to reach {int(new_prob*100)}% approval probability"
                )

        # 🔹 INCOME
        elif feature == "income_annum":
            new_val, new_prob = simulate_feature_change(
                feature, current_val, step=50000, direction="increase"
            )
            if new_val:
                suggestions.append(
                    f"Increase income from ₹{int(current_val):,} → ₹{int(new_val):,} to reach {int(new_prob*100)}% approval probability"
                )

        # 🔹 LOAN AMOUNT
        elif feature == "loan_amount":
            new_val, new_prob = simulate_feature_change(
                feature, current_val, step=100000, direction="decrease"
            )
            if new_val:
                suggestions.append(
                    f"Reduce loan amount from ₹{int(current_val):,} → ₹{int(new_val):,} to reach {int(new_prob*100)}% approval probability"
                )

        # 🔹 LOAN TERM
        elif feature == "loan_term":
            new_val, new_prob = simulate_feature_change(
                feature, current_val, step=1, direction="decrease"
            )
            if new_val:
                suggestions.append(
                    f"Reduce loan term from {int(current_val)} → {int(new_val)} years to reach {int(new_prob*100)}% approval probability"
                )

        # 🔹 GENERIC FALLBACK
        else:
            suggestions.append(f"Improve {feature.replace('_', ' ')}")

    # -------- FINAL RETURN --------
    return pred, prob, suggestions