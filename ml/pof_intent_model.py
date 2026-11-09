"""
Proof of Future: alış niyyəti modeli (təkrar işlədilə bilən öyrətmə skripti)

Data: UCI "Online Shoppers Purchasing Intention" (12 330 real e-ticarət sessiyası)
  https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Model: logistic regression (demoda brauzerdə işləyir) + müqayisə üçün gradient boosting.

İşlətmək:
  pip install pandas scikit-learn
  python pof_intent_model.py online_shoppers_intention.csv
Nəticə: metrikalar ekrana çıxır, model çəkiləri pof_model.json faylına yazılır.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLS = ["admin_n", "admin_s", "info_n", "info_s", "prod_n", "prod_s",
        "new_visitor", "weekend", "bounce", "exit", "pagevalue"]


def features(d: pd.DataFrame) -> pd.DataFrame:
    """SDK-nın sessiya ərzində hesablaya bildiyi xüsusiyyətlər.
    bounce, exit və pagevalue səhifə səviyyəsində aqreqat cədvəldən gəlir (şəxsi data deyil)."""
    return pd.DataFrame({
        "admin_n": np.log1p(d["Administrative"]),
        "admin_s": np.log1p(d["Administrative_Duration"]),
        "info_n": np.log1p(d["Informational"]),
        "info_s": np.log1p(d["Informational_Duration"]),
        "prod_n": np.log1p(d["ProductRelated"]),
        "prod_s": np.log1p(d["ProductRelated_Duration"]),
        "new_visitor": (d["VisitorType"] == "New_Visitor").astype(int),
        "weekend": d["Weekend"].astype(int),
        "bounce": d["BounceRates"],
        "exit": d["ExitRates"],
        "pagevalue": np.log1p(d["PageValues"]),
    })[COLS]


def main(path: str) -> None:
    df = pd.read_csv(path)
    y = df["Revenue"].astype(int).values
    X = features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000).fit(scaler.transform(Xtr), ytr)
    p = lr.predict_proba(scaler.transform(Xte))[:, 1]

    gbm = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7).fit(Xtr, ytr)
    pg = gbm.predict_proba(Xte)[:, 1]

    top = np.argsort(-p)[: len(p) // 10]
    frac, mean_pred = calibration_curve(yte, p, n_bins=5, strategy="quantile")
    metrics = {
        "sessions_total": int(len(df)),
        "sessions_test": int(len(Xte)),
        "purchase_rate": round(float(y.mean()), 4),
        "auc_logreg": round(roc_auc_score(yte, p), 3),
        "auc_gradient_boosting": round(roc_auc_score(yte, pg), 3),
        "average_precision": round(average_precision_score(yte, p), 3),
        "brier": round(brier_score_loss(yte, p), 4),
        "brier_baseline": round(brier_score_loss(yte, np.full_like(p, ytr.mean())), 4),
        "top10_purchase_rate": round(float(yte[top].mean()), 3),
        "calibration_pred_vs_real": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(mean_pred, frac)],
    }

    # Brauzer üçün: standartlaşdırmanı çəkilərə daxil edirik (xam xüsusiyyətlər üzərində işləyir)
    w = lr.coef_[0] / scaler.scale_
    b = lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_)
    model = {"cols": COLS, "w": [round(float(v), 5) for v in w], "b": round(float(b), 5),
             "standardized_coef": {c: round(float(v), 3) for c, v in zip(COLS, lr.coef_[0])}}

    with open("pof_model.json", "w") as fh:
        json.dump({"metrics": metrics, "model": model}, fh, indent=1)
    print(json.dumps(metrics, indent=1))
    print("Çəkilər pof_model.json faylına yazıldı.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "online_shoppers_intention.csv")
