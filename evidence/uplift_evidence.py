"""
Proof of Future: əsas fərziyyənin real data ilə yoxlanılması

Sual: səbəbə uyğun tədbir hamıya eyni tədbirdən daha çox qazandırırmı?

Data: Kevin Hillstrom, MineThatData E-Mail Analytics Challenge (2008).
  64 000 müştəri təsadüfi olaraq 3 qrupa bölünüb: kişi kataloqu e-poçtu,
  qadın kataloqu e-poçtu, e-poçt yoxdur (nəzarət). 2 həftə ərzində alış və xərc ölçülüb.
  http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Metod:
  1. Orta təsir (ATE): e-poçt qrupları vs nəzarət, 95% etibar intervalı.
  2. Heterojenlik: hər e-poçtun təsiri müştərinin əvvəl nə aldığına görə.
  3. Siyasətlərin dəyəri: təsadüfi bölgü olduğu üçün inverse propensity weighting (p = 1/3)
     ilə qərəzsiz qiymətləndirmə. "Səbəbə uyğun" qayda əvvəlcədən sabitdir (datadan öyrənilmir),
     ona görə overfitting yoxdur. Etibar intervalları 1000 bootstrap ilə.
  Pul: marja 25%, alış edən hər müştəriyə 10% kupon xərci (ROI hesablayıcısı ilə eyni fərziyyələr).

İşlətmək: python uplift_evidence.py hillstrom.csv
Nəticə: evidence.json və evidence_chart.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MARGIN, COUPON, B, SEED = 0.25, 0.10, 1000, 0


def load(path):
    d = pd.read_csv(path)
    d["arm"] = d.segment.map({"No E-Mail": "none", "Mens E-Mail": "mens", "Womens E-Mail": "womens"})
    d["buyer"] = np.where((d.mens == 1) & (d.womens == 1), "both", np.where(d.mens == 1, "mens_only", "womens_only"))
    # Səbəbə uyğun qayda: müştəriyə əvvəl aldığı kateqoriyanın kataloqu göndərilir
    d["matched"] = np.where(d.mens == 1, "mens", "womens")
    return d


def diff_ci(a, b):
    m = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return m, m - 1.96 * se, m + 1.96 * se


def policy_value(d, choice):
    """1000 müştəriyə xalis marja (kupon xərci çıxılmaqla), IPW ilə."""
    hit = (d.arm.values == choice) * 3.0
    treated = choice != "none"
    net = d.spend.values * MARGIN - COUPON * d.spend.values * treated
    return (hit * net).mean() * 1000


def main(path):
    d = load(path)
    rng = np.random.RandomState(SEED)
    treated = d[d.arm != "none"]
    control = d[d.arm == "none"]

    ate = {}
    for y in ["visit", "conversion", "spend"]:
        m, lo, hi = diff_ci(treated[y], control[y])
        base = control[y].mean()
        ate[y] = {"base": round(base, 5), "uplift": round(m, 5), "ci": [round(lo, 5), round(hi, 5)],
                  "relative_pct": round(100 * m / base, 1), "relative_ci_pct": [round(100 * lo / base, 1), round(100 * hi / base, 1)]}

    het = []
    for buyer, g in d.groupby("buyer"):
        base = g[g.arm == "none"]
        for arm in ["mens", "womens"]:
            m, lo, hi = diff_ci(g[g.arm == arm].conversion, base.conversion)
            het.append({"buyer": buyer, "email": arm, "conv_uplift_pp": round(100 * m, 3),
                        "ci_pp": [round(100 * lo, 3), round(100 * hi, 3)]})

    policies = {
        "none": lambda x: np.array(["none"] * len(x)),
        "blanket_wrong": lambda x: np.array(["womens"] * len(x)),
        "blanket_random": lambda x: np.where(np.random.RandomState(1).rand(len(x)) < 0.5, "mens", "womens"),
        "blanket_best": lambda x: np.array(["mens"] * len(x)),
        "matched": lambda x: x["matched"].values,
    }
    point = {k: policy_value(d, f(d)) for k, f in policies.items()}
    boots = {k: [] for k in policies}
    for _ in range(B):
        s = d.sample(len(d), replace=True, random_state=rng)
        for k, f in policies.items():
            boots[k].append(policy_value(s, f(s)))
    boots = {k: np.array(v) for k, v in boots.items()}
    gain = {}
    for k in policies:
        if k == "none":
            continue
        diff = boots[k] - boots["none"]
        gain[k] = {"vs_none": round(point[k] - point["none"], 1),
                   "ci": [round(np.percentile(diff, 2.5), 1), round(np.percentile(diff, 97.5), 1)],
                   "p_positive": round(float((diff > 0).mean()), 3)}
    md = boots["matched"] - boots["blanket_wrong"]
    matched_vs_wrong = {"diff": round(point["matched"] - point["blanket_wrong"], 1),
                        "ci": [round(np.percentile(md, 2.5), 1), round(np.percentile(md, 97.5), 1)]}

    out = {"n": int(len(d)), "margin": MARGIN, "coupon": COUPON, "bootstrap": B, "ate": ate,
           "heterogeneity": het, "policy_net_per_1000": {k: round(v, 1) for k, v in point.items()},
           "gain_vs_none_per_1000": gain, "matched_vs_wrong_per_1000": matched_vs_wrong}
    json.dump(out, open("evidence.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    chart(gain, het)


def chart(gain, het):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor("white")
    ink, muted, grid, accent, gray = "#1b2230", "#5b6575", "#e6e9ef", "#0f9f7f", "#a9b1bd"

    order = [("blanket_wrong", "Hamıya yanlış tədbir\n(qadın kataloqu)"),
             ("blanket_random", "Hamıya təsadüfi tədbir"),
             ("blanket_best", "Hamıya ən yaxşı tədbir\n(sonradan bilinir)"),
             ("matched", "Səbəbə uyğun tədbir")]
    ys = np.arange(len(order))[::-1]
    for y, (k, lbl) in zip(ys, order):
        g = gain[k]
        col = accent if k == "matched" else gray
        a1.barh(y, g["vs_none"], height=0.5, color=col, zorder=3)
        a1.errorbar(g["vs_none"], y, xerr=[[g["vs_none"] - g["ci"][0]], [g["ci"][1] - g["vs_none"]]],
                    fmt="none", ecolor=ink, elinewidth=1.4, capsize=5, zorder=4)
        a1.text(max(g["ci"][1], 0) + 4, y, f'{g["vs_none"]:+.0f} $', va="center", color=ink,
                fontsize=14, fontweight="bold" if k == "matched" else "normal")
    a1.set_yticks(ys, [l for _, l in order], color=ink)
    a1.axvline(0, color=muted, linewidth=1)
    a1.set_xlabel("Tədbirsizliyə nisbətən xalis əlavə marja, 1000 müştəriyə ($)\nmarja 25%, alış edənə 10% kupon xərci", color=muted)
    a1.set_title("Yanlış tədbir pul qazandırmır,\nsəbəbə uyğun tədbir qazandırır", loc="left", fontsize=17, fontweight="bold", color=ink)
    a1.grid(axis="x", color=grid, zorder=0)
    a1.set_xlim(-60, 125)

    rows = [r for r in het if r["email"] == "womens" and r["buyer"] in ("womens_only", "mens_only")]
    names = {"womens_only": "Qadın məhsulu alanlar", "mens_only": "Kişi məhsulu alanlar"}
    ys2 = np.arange(len(rows))[::-1]
    for y, r in zip(ys2, rows):
        v = r["conv_uplift_pp"]
        col = accent if r["buyer"] == "womens_only" else gray
        a2.barh(y, v, height=0.45, color=col, zorder=3)
        a2.errorbar(v, y, xerr=[[v - r["ci_pp"][0]], [r["ci_pp"][1] - v]], fmt="none", ecolor=ink, elinewidth=1.4, capsize=5, zorder=4)
        a2.text(r["ci_pp"][1] + 0.03, y, f"{v:+.2f} f.b.", va="center", color=ink, fontsize=14)
    a2.set_yticks(ys2, [names[r["buyer"]] for r in rows], color=ink)
    a2.axvline(0, color=muted, linewidth=1)
    a2.set_xlabel("Alış ehtimalının artımı, faiz bəndi (95% etibar intervalı)", color=muted)
    a2.set_title("Eyni tədbir (qadın kataloqu),\nfərqli müştəri, fərqli nəticə", loc="left", fontsize=17, fontweight="bold", color=ink)
    a2.grid(axis="x", color=grid, zorder=0)
    a2.set_xlim(-0.3, 1.0)

    for ax in (a1, a2):
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, length=0)
    fig.text(0.01, 0.01, "Data: Hillstrom MineThatData e-poçt eksperimenti, 64 000 müştəri, təsadüfi bölgü. "
             "Siyasət dəyəri IPW ilə, intervallar 1000 bootstrap ilə.", color=muted, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("evidence_chart.png", dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hillstrom.csv")
