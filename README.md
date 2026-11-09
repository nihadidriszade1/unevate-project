# Proof of Future: live demo and evidence pack

A hackathon demo of a retention SDK that understands why a customer is about to leave before they do, and removes that cause instead of handing out a coupon.

## What's in the package

| Folder | Contents |
|---|---|
| `demo/` | Live demo: `index.html` and 6 product images. Works offline. |
| `ml/` | Purchase-intent model: training script (`pof_intent_model.py`) and weights (`pof_model.json`) |
| `evidence/` | Evidence for the core hypothesis: analysis script, results (`evidence.json`), slide chart (`evidence_chart.png`) |
| `video/` | Backup demo videos, 1080p, about 75 seconds each, with captions in Azerbaijani and English |
| `get_data.sh` | Downloads the two open datasets used in the analyses |

## Running the demo

Open `demo/index.html` in Chrome. No server is needed.

- Without internet, Google Fonts won't load and system fonts are used instead. Everything else works.
- Online version: https://claude.ai/artifact/XgKPTTaD2gDzkQDoeVUwiu
- Best viewed at 1366×768 or larger, with the browser in full-screen mode (F11).

### Keyboard shortcuts

| Key | What it does |
|---|---|
| `1` `2` `3` `4` | Start a scenario: price, payment, size, delay |
| `R` | Restart the current scenario |
| `P` | Switch between Pitch and Detail mode |
| `L` | Language: AZ → TR → EN |
| `E` | ROI calculator: 100 at-risk customers |
| `M` | Sources: where every number comes from, plus the experimental evidence |
| `Esc` | Close the open panel |

### Scenarios

| # | What the judge does | What the system does |
|---|---|---|
| 1 | Taps the price 3 times | Price hesitation → price shield |
| 1 + "Bargain hunter" profile | The same 3 taps | "Normal behaviour": no intervention |
| 2 | Taps "Pay" repeatedly | Technical friction → one-tap payment |
| 3 | Opens the size chart, changes size twice | Size doubt → size assistant |
| 4 | Nothing | Delayed order → apology and compensation before a complaint |

In scenario 3 the size buttons may sit at the bottom of the screen. Scroll the phone screen down a little.

### Recommended pitch flow (about 2 minutes)

1. `1`: the judge taps the price 3 times. The decision appears within milliseconds, and the money card shows the cost and the net gain.
2. Switch the profile to "Bargain hunter": the same behaviour is treated as normal.
3. `2`: the judge taps "Pay" in frustration.
4. `M`: "We don't assume this, we measured it": the 64,000-customer experiment.

## Key numbers and their sources

### ML model (purchase intent)

- **Data:** UCI Online Shoppers Purchasing Intention, 12,330 real e-commerce sessions (CC BY 4.0).
- **Result:** AUC **0.889** on 2,466 held-out sessions (0.903 with gradient boosting).
- **Impact:** the top 10% of sessions convert at 70%, against an average of 15.5%.
- **Reproduce:** `python ml/pof_intent_model.py data/online_shoppers_intention.csv`

### Evidence: matching the action to the cause

- **Data:** Hillstrom MineThatData e-mail experiment. 64,000 customers randomly split into 3 groups.
- **Average treatment effect:** purchases **+87%** (95% interval: +62% … +111%).
- **Same action, different customer:** the women's catalogue lifted women buyers by **+0.51 percentage points** and men buyers by +0.06 percentage points (statistically zero).
- **Net margin per 1,000 customers** (25% margin, 10% coupon for buyers):
  - wrong action for everyone: **−$1**
  - action matched to the cause: **+$52**. This is the only policy whose interval lies above zero.
- **Reproduce:** `python evidence/uplift_evidence.py data/hillstrom.csv`

### Still hypotheses (to be measured in the pilot)

- **Hypotheses:** the cause threshold (0.62), signal half-lives, trust changes, the cost of the shield, margin and 90-day customer value, and the share of customers in the hesitation zone (60%).
- **All visible in the demo:** the Sources panel shows each one's status and how it will be measured in the pilot.
- **Honest limitations:**
  - The evidence experiment is in an e-mail context, not an in-app offer.
  - The "which action" question is proven; the "to whom" question is not yet proven.
  - The cause layer still runs on transparent rules, because no open dataset with cause labels exists.

## Reproducing the analyses

```bash
./get_data.sh
pip install pandas scikit-learn matplotlib
python ml/pof_intent_model.py data/online_shoppers_intention.csv
python evidence/uplift_evidence.py data/hillstrom.csv
```

## Sources

- UCI Online Shoppers Purchasing Intention Dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Hillstrom MineThatData E-Mail Analytics: http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- Rage click definition (Contentsquare): https://support.contentsquare.com/hc/en-us/articles/37271767248273-Frustration-score
- Baumeister et al. (2001), Bad Is Stronger Than Good: https://fbaum.unc.edu/teaching/articles/Baumeister_2001.pdf
- Kahneman et al. (1993), When More Pain Is Preferred to Less: https://www.jstor.org/stable/40062570
- Product images: Unsplash (https://unsplash.com)
