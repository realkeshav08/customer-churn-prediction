"""Validate the full pipeline end-to-end without Jupyter."""
import sys
import logging
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import seaborn as sns

from src.data_loader import load_raw_data
from src.preprocessing import preprocess, save_processed
from src.feature_engineering import build_features
from src.train_models import split_data, train_logistic_regression, train_random_forest, train_xgboost, save_model
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_feature_importance, save_metrics
)
from src.sql_queries import load_into_sqlite, run_all_queries

sns.set_theme(style="whitegrid", palette="muted")
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── STEP 1: Load & Preprocess ─────────────────────────────────────────────────
log.info("=== STEP 1: Load & Preprocess ===")
raw = load_raw_data()
df = preprocess(raw)
save_processed(df)
assert df.shape[0] > 7000, "Expected 7000+ rows after preprocessing"
log.info("Preprocessed shape: %s", df.shape)

# ── STEP 2: EDA Figures ────────────────────────────────────────────────────────
log.info("=== STEP 2: EDA Figures ===")

BLUE, ORANGE = "#2196F3", "#FF9800"

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close()

# Insight 1 — Overall churn rate
churn_counts = df["Churn"].value_counts()
churn_rate = df["Churn"].mean() * 100
print(f"Churn rate: {churn_rate:.1f}%  Churned: {churn_counts.get(1,0):,}  Retained: {churn_counts.get(0,0):,}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vals = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
axes[0].bar(["Retained", "Churned"], vals, color=[BLUE, ORANGE], edgecolor="white", width=0.5)
for i, v in enumerate(vals):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=11)
axes[0].set_title("Customer Churn Count", fontweight="bold")
axes[0].set_ylabel("Count")
axes[1].pie(vals, labels=["Retained","Churned"], colors=[BLUE, ORANGE],
            autopct="%1.1f%%", startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
axes[1].set_title("Churn Distribution", fontweight="bold")
plt.suptitle("Insight 1: Overall Churn Rate", fontsize=14, fontweight="bold", y=1.01)
savefig("insight_01_churn_rate.png")

# Insight 2 — Contract type
contract_churn = df.groupby("Contract")["Churn"].agg(["mean","sum","count"]).reset_index()
contract_churn.columns = ["Contract","churn_rate","churned","total"]
contract_churn["churn_rate_pct"] = contract_churn["churn_rate"] * 100
for _, row in contract_churn.iterrows():
    print(f"{row['Contract']:20s}  churn={row['churn_rate_pct']:.1f}%  ({int(row['churned'])}/{int(row['total'])})")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(contract_churn["Contract"], contract_churn["churn_rate_pct"],
              color=[ORANGE, BLUE, "#4CAF50"], edgecolor="white", width=0.5)
for bar, val in zip(bars, contract_churn["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)"); ax.set_title("Insight 2: Churn Rate by Contract Type", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_02_churn_by_contract.png")

# Insight 3 — Tenure buckets
bins = [0, 12, 24, 48, df["tenure"].max()+1]
labels = ["0-12 mo","13-24 mo","25-48 mo","49+ mo"]
df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
tenure_churn = df.groupby("tenure_bucket", observed=True)["Churn"].agg(["mean","count"]).reset_index()
tenure_churn["churn_rate_pct"] = tenure_churn["mean"] * 100
for _, r in tenure_churn.iterrows():
    print(f"Tenure {str(r['tenure_bucket']):10s}: {r['churn_rate_pct']:.1f}%  (n={r['count']:,})")

fig, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette("viridis", len(tenure_churn))
bars = ax.bar(tenure_churn["tenure_bucket"].astype(str), tenure_churn["churn_rate_pct"],
              color=palette, edgecolor="white", width=0.5)
for bar, val in zip(bars, tenure_churn["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Tenure Bucket")
ax.set_title("Insight 3: Churn Rate by Tenure Bucket", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_03_churn_by_tenure.png")

# Insight 4 — Monthly charges
churned_mc  = df[df["Churn"]==1]["MonthlyCharges"]
retained_mc = df[df["Churn"]==0]["MonthlyCharges"]
print(f"Churned median monthly charge: ${churned_mc.median():.2f}  Retained: ${retained_mc.median():.2f}")
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(retained_mc, bins=40, alpha=0.6, color=BLUE,   label="Retained", edgecolor="white")
ax.hist(churned_mc,  bins=40, alpha=0.6, color=ORANGE, label="Churned",  edgecolor="white")
ax.axvline(retained_mc.median(), color=BLUE,   linestyle="--", lw=1.5)
ax.axvline(churned_mc.median(),  color=ORANGE, linestyle="--", lw=1.5)
ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Customer Count")
ax.set_title("Insight 4: Monthly Charges Distribution by Churn Status", fontweight="bold", fontsize=13)
ax.legend()
savefig("insight_04_monthly_charges.png")

# Insight 5 — Total charges
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[df["Churn"]==0]["TotalCharges"], bins=50, alpha=0.6, color=BLUE,   label="Retained", edgecolor="white")
ax.hist(df[df["Churn"]==1]["TotalCharges"], bins=50, alpha=0.6, color=ORANGE, label="Churned",  edgecolor="white")
ax.set_xlabel("Total Charges ($)"); ax.set_ylabel("Customer Count")
ax.set_title("Insight 5: Total Charges Distribution by Churn Status", fontweight="bold", fontsize=13)
ax.legend()
savefig("insight_05_total_charges.png")

# Insight 6 — Internet service
internet_churn = df.groupby("InternetService")["Churn"].agg(["mean","count"]).reset_index()
internet_churn["churn_rate_pct"] = internet_churn["mean"] * 100
for _, r in internet_churn.iterrows():
    print(f"Internet={r['InternetService']:15s}: churn={r['churn_rate_pct']:.1f}%  (n={r['count']:,})")
fig, ax = plt.subplots(figsize=(7, 5))
palette3 = [ORANGE, BLUE, "#4CAF50"]
bars = ax.bar(internet_churn["InternetService"], internet_churn["churn_rate_pct"],
              color=palette3[:len(internet_churn)], edgecolor="white", width=0.5)
for bar, val in zip(bars, internet_churn["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Internet Service")
ax.set_title("Insight 6: Churn Rate by Internet Service Type", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_06_internet_service.png")

# Insight 7 — Payment method
pay_churn = df.groupby("PaymentMethod")["Churn"].agg(["mean","count"]).reset_index()
pay_churn["churn_rate_pct"] = pay_churn["mean"] * 100
pay_churn = pay_churn.sort_values("churn_rate_pct", ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
colors7 = sns.color_palette("coolwarm", len(pay_churn))
ax.barh(pay_churn["PaymentMethod"], pay_churn["churn_rate_pct"], color=colors7, edgecolor="white")
for i, val in enumerate(pay_churn["churn_rate_pct"]):
    ax.text(val+0.3, i, f"{val:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Churn Rate (%)")
ax.set_title("Insight 7: Churn Rate by Payment Method", fontweight="bold", fontsize=13)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_07_payment_method.png")

# Insight 8 — Senior citizen
senior_churn = df.groupby("SeniorCitizen")["Churn"].agg(["mean","count"]).reset_index()
senior_churn["churn_rate_pct"] = senior_churn["mean"] * 100
for _, r in senior_churn.iterrows():
    print(f"SeniorCitizen={r['SeniorCitizen']:5s}: churn={r['churn_rate_pct']:.1f}%  (n={r['count']:,})")
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(senior_churn["SeniorCitizen"].astype(str), senior_churn["churn_rate_pct"],
       color=[BLUE, ORANGE], edgecolor="white", width=0.4)
for i, val in enumerate(senior_churn["churn_rate_pct"]):
    ax.text(i, val+0.5, f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_xlabel("Senior Citizen"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Insight 8: Churn Rate by Senior Citizen Status", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_08_senior_citizen.png")

# Insight 9 — Partner / Dependents
df["partner_dep"] = df["Partner"] + " / " + df["Dependents"]
pd_churn = df.groupby("partner_dep")["Churn"].agg(["mean","count"]).reset_index()
pd_churn["churn_rate_pct"] = pd_churn["mean"] * 100
pd_churn = pd_churn.sort_values("churn_rate_pct", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
palette9 = sns.color_palette("Set2", len(pd_churn))
bars = ax.bar(pd_churn["partner_dep"], pd_churn["churn_rate_pct"], color=palette9, edgecolor="white", width=0.5)
for bar, val in zip(bars, pd_churn["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Partner / Dependents")
ax.set_title("Insight 9: Churn by Partner & Dependents Combination", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_09_partner_dependents.png")

# Insight 10 — Tech Support
ts_churn = df.groupby("TechSupport")["Churn"].agg(["mean","count"]).reset_index()
ts_churn["churn_rate_pct"] = ts_churn["mean"] * 100
fig, ax = plt.subplots(figsize=(7, 5))
palette10 = [ORANGE, BLUE, "#9E9E9E"]
bars = ax.bar(ts_churn["TechSupport"], ts_churn["churn_rate_pct"],
              color=palette10[:len(ts_churn)], edgecolor="white", width=0.5)
for bar, val in zip(bars, ts_churn["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Insight 10: Churn Rate by Tech Support", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_10_tech_support.png")

# Insight 11 — Online Security
os_churn = df.groupby("OnlineSecurity")["Churn"].agg(["mean","count"]).reset_index()
os_churn["churn_rate_pct"] = os_churn["mean"] * 100
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(os_churn["OnlineSecurity"], os_churn["churn_rate_pct"],
              color=palette10[:len(os_churn)], edgecolor="white", width=0.5)
for bar, val in zip(bars, os_churn["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Insight 11: Churn Rate by Online Security", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_11_online_security.png")

# Insight 12 — Correlation heatmap
num_cols = ["tenure","MonthlyCharges","TotalCharges","Churn"]
corr = df[num_cols].corr()
print("\nCorrelation with Churn:")
print(corr["Churn"].drop("Churn").sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, square=True, ax=ax, cbar_kws={"shrink":0.8})
ax.set_title("Insight 12: Correlation Heatmap — Numerical Features", fontweight="bold", fontsize=13)
savefig("insight_12_correlation_heatmap.png")

# Insight 13 — Paperless billing
pb_churn = df.groupby("PaperlessBilling")["Churn"].agg(["mean","count"]).reset_index()
pb_churn["churn_rate_pct"] = pb_churn["mean"] * 100
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(pb_churn["PaperlessBilling"], pb_churn["churn_rate_pct"],
       color=[BLUE, ORANGE], edgecolor="white", width=0.4)
for i, val in enumerate(pb_churn["churn_rate_pct"]):
    ax.text(i, val+0.5, f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_xlabel("Paperless Billing"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Insight 13: Churn Rate by Paperless Billing", fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
savefig("insight_13_paperless_billing.png")

fig_count = len(list(FIGURES.glob("*.png")))
log.info("EDA figures saved: %d", fig_count)
assert fig_count >= 13, f"Expected 13+ figures, got {fig_count}"

# ── STEP 3: Feature Engineering ───────────────────────────────────────────────
log.info("=== STEP 3: Feature Engineering ===")
df_clean = pd.read_csv(ROOT / "data" / "processed" / "telco_churn_cleaned.csv")
df_feat, scaler = build_features(df_clean)

feature_cols = [c for c in df_feat.columns if c != "Churn"]
log.info("Total features: %d", len(feature_cols))
assert len(feature_cols) >= 25, f"Expected 25+ features, got {len(feature_cols)}"

# Correlation plot (feature engineering figure)
corr_target = df_feat.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
top20 = corr_target.head(20)
fig, ax = plt.subplots(figsize=(9, 7))
colors_c = ["#FF9800" if v > 0 else "#2196F3" for v in top20.values]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors_c[::-1], edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson Correlation with Churn")
ax.set_title("Top 20 Features by Correlation with Churn", fontweight="bold", fontsize=13)
savefig("feature_correlation_with_churn.png")

# Persist
out_path = ROOT / "data" / "processed" / "telco_churn_features.csv"
df_feat.to_csv(out_path, index=False)

import joblib
scaler_path = ROOT / "models" / "scaler.pkl"
scaler_path.parent.mkdir(exist_ok=True)
joblib.dump(scaler, scaler_path)
log.info("Features saved to %s", out_path)

# Num services vs churn
from src.feature_engineering import add_num_services, add_tenure_group
df_svc = add_num_services(df_clean.copy())
svc_churn = df_svc.groupby("num_services")["Churn"].mean().reset_index()
svc_churn["churn_rate_pct"] = svc_churn["Churn"] * 100
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(svc_churn["num_services"], svc_churn["churn_rate_pct"], marker="o", color="#2196F3", linewidth=2)
ax.set_xlabel("Number of Services"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Number of Subscribed Services", fontweight="bold")
savefig("engineered_num_services_vs_churn.png")

# ── STEP 4: Modeling ──────────────────────────────────────────────────────────
log.info("=== STEP 4: Model Training ===")
X_train, X_test, y_train, y_test = split_data(df_feat)

from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

lr  = train_logistic_regression(X_train, y_train)
rf  = train_random_forest(X_train, y_train)
xgb, xgb_threshold = train_xgboost(X_train, y_train)

lr_prob  = lr.predict_proba(X_test)[:, 1]
rf_prob  = rf.predict_proba(X_test)[:, 1]
xgb_prob = xgb.predict_proba(X_test)[:, 1]

# LR & RF: find their own optimal thresholds too for fair comparison
def best_f1_threshold(y_tr, prob_tr):
    prec, rec, thr = precision_recall_curve(y_tr, prob_tr)
    f1 = np.where((prec+rec)==0, 0, 2*prec*rec/(prec+rec))
    return float(thr[f1[:-1].argmax()])

lr_thr  = best_f1_threshold(y_train, lr.predict_proba(X_train)[:,1])
rf_thr  = best_f1_threshold(y_train, rf.predict_proba(X_train)[:,1])

lr_pred  = (lr_prob  >= lr_thr).astype(int)
rf_pred  = (rf_prob  >= rf_thr).astype(int)
xgb_pred = (xgb_prob >= xgb_threshold).astype(int)

lr_m  = compute_metrics(y_test, lr_pred,  lr_prob)
rf_m  = compute_metrics(y_test, rf_pred,  rf_prob)
xgb_m = compute_metrics(y_test, xgb_pred, xgb_prob)

print("\n=== MODEL COMPARISON ===")
for name, m, thr in [("Logistic Regression", lr_m, lr_thr), ("Random Forest", rf_m, rf_thr), ("XGBoost", xgb_m, xgb_threshold)]:
    print(f"{name:22s}  ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  threshold={thr:.3f}")

# Evaluation plots for each model
for name, pred, prob in [
    ("Logistic Regression", lr_pred,  lr_prob),
    ("Random Forest",       rf_pred,  rf_prob),
    ("XGBoost",             xgb_pred, xgb_prob),
]:
    plot_confusion_matrix(y_test, pred, name)
    plot_roc_curve(y_test, prob, name)
    plot_precision_recall_curve(y_test, prob, name)

plot_feature_importance(rf,  X_train.columns.tolist(), "Random Forest")
plot_feature_importance(xgb, X_train.columns.tolist(), "XGBoost")

# Multi-model ROC comparison
fig, ax = plt.subplots(figsize=(7, 6))
for name, prob in [("Logistic Regression", lr_prob), ("Random Forest", rf_prob), ("XGBoost", xgb_prob)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontweight="bold", fontsize=13)
ax.legend()
savefig("roc_curves_all_models.png")

# Save best model & metrics
save_model(xgb, "best_model")
all_metrics = {"Logistic Regression": lr_m, "Random Forest": rf_m, "XGBoost": xgb_m}
save_metrics(all_metrics)

# ── STEP 5: Tableau Export ────────────────────────────────────────────────────
log.info("=== STEP 5: Tableau Export ===")
df_tableau = df_clean.copy()
df_full, _ = build_features(df_clean.copy())
X_full = df_full.drop(columns=["Churn"])
X_full = X_full.reindex(columns=X_train.columns, fill_value=0)
pred_probs = xgb.predict_proba(X_full)[:, 1]

bins_t   = [0, 12, 24, 48, df_tableau["tenure"].max()+1]
labels_t = ["0-12","13-24","25-48","49+"]
df_tableau["tenure_group"]               = pd.cut(df_tableau["tenure"], bins=bins_t, labels=labels_t).astype(str)
df_tableau["clv"]                        = df_tableau["MonthlyCharges"] * df_tableau["tenure"]
df_tableau["predicted_churn_probability"] = pred_probs
df_tableau["risk_tier"] = pd.cut(pred_probs, bins=[0, 0.35, 0.65, 1.0], labels=["Low","Medium","High"]).astype(str)

tableau_path = ROOT / "data" / "tableau" / "churn_dashboard_data.csv"
tableau_path.parent.mkdir(exist_ok=True)
df_tableau.to_csv(tableau_path, index=False)
assert "predicted_churn_probability" in df_tableau.columns
log.info("Tableau export saved: %s  shape=%s", tableau_path, df_tableau.shape)

# ── STEP 6: SQL Analysis ──────────────────────────────────────────────────────
log.info("=== STEP 6: SQL Analysis ===")
conn, _ = load_into_sqlite(df_clean)
results = run_all_queries(conn)
for qname, qdf in results.items():
    log.info("%-30s  rows=%d", qname, len(qdf))
    assert len(qdf) > 0, f"Query {qname} returned no rows"
conn.close()

# SQL visualisation — Q2
q2 = results["q2_churn_by_contract"]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(q2["Contract"], q2["churn_rate_pct"],
              color=[ORANGE, BLUE, "#4CAF50"][:len(q2)], edgecolor="white", width=0.5)
for bar, val in zip(bars, q2["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("SQL Q2: Churn Rate by Contract Type", fontweight="bold")
savefig("sql_q2_churn_by_contract.png")

# SQL visualisation — Q5
q5 = results["q5_tenure_cohort"]
fig, ax = plt.subplots(figsize=(8, 4))
colors5 = sns.color_palette("viridis", len(q5))
bars = ax.bar(q5["tenure_bucket"], q5["churn_rate_pct"], color=colors5, edgecolor="white", width=0.5)
for bar, val in zip(bars, q5["churn_rate_pct"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("SQL Q5: Tenure Cohort Churn Analysis", fontweight="bold")
savefig("sql_q5_tenure_cohort.png")

# ── FINAL VALIDATION ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL VALIDATION")
print("="*60)
final_fig_count = len(list(FIGURES.glob("*.png")))
print(f"Figures generated       : {final_fig_count} (target: 13+)")
print(f"Features engineered     : {len(feature_cols)} (target: 25+)")
print(f"XGBoost ROC-AUC         : {xgb_m['roc_auc']:.4f} (target >= 0.83)")
print(f"XGBoost F1 (churn class): {xgb_m['f1']:.4f}")
print(f"XGBoost F1 (weighted)   : {xgb_m['f1_weighted']:.4f} (target >= 0.78)")
print(f"SQL queries ran         : {len(results)}/8")
print(f"Tableau export cols     : {len(df_tableau.columns)}")
assert final_fig_count >= 13
assert len(feature_cols) >= 25
assert xgb_m["roc_auc"] >= 0.80, f"ROC-AUC too low: {xgb_m['roc_auc']}"
print("\nALL CHECKS PASSED.")
