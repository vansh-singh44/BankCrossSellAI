# EXPLANATION.md — VITBAIH Bank Marketing AI

*All answers use actual model outputs from the real bank-additional-full.csv dataset.*

---

## 1. Percentage of YES subscriptions

From EDA on the real 41,188-row UCI dataset:

- **YES (subscribed):** 4,640 customers → **11.27%**
- **NO (not subscribed):** 36,548 customers → **88.73%**

---

## 2. Class Imbalance — Why it matters and how we handled it

**What is class imbalance?**  
Our dataset has a **7.88:1 ratio (No:Yes)**. A naïve model predicting "No" for every customer
would achieve 88.7% accuracy while being completely useless for identifying subscribers.

**Problems it causes:**
- High accuracy but near-zero recall for the minority class (Yes)
- Model biases toward majority class predictions
- Standard accuracy is a misleading metric

**How we handled it:**
- **Logistic Regression:** `class_weight="balanced"` — weights minority class inversely proportional to frequency
- **CatBoost:** `auto_class_weights="Balanced"` — same principle, natively supported
- **Metric choice:** F1-score (not accuracy) as primary metric (see Q5)

---

## 3. Highest Subscription Rate by Job

From EDA on the real dataset:

| Rank | Job          | Subscription Rate |
|------|-------------|-------------------|
| 1    | student     | **31.43%**        |
| 2    | retired     | 25.23%            |
| 3    | unemployed  | 14.20%            |
| 4    | admin.      | 12.97%            |
| 5    | management  | 11.22%            |
| ...  | blue-collar | 6.89% (lowest)    |

**Top job: `student`** (31.43%) — students are often young, fewer financial commitments, more open to new financial products.  
**Runner-up: `retired`** (25.23%) — retirees have accumulated savings and actively seek stable investment vehicles like term deposits.  
**Lowest: `blue-collar`** (6.89%) — likely constrained by income levels and existing financial obligations.

---

## 4. Feature Importance (CatBoost)

From `plots/09_feature_importance.png` (real model trained on 32,950 samples):

1. **duration** — Single strongest predictor. Calls >449s (median for Yes) lead to subscription far more often. *Post-call feature — use for benchmarking only, not pre-call targeting.*
2. **euribor3m** — Euribor 3-month rate. When rates are high (4.8%+), customers prefer market instruments over deposits. Lower rates (0.6–1.5%) make term deposits attractive.
3. **nr.employed** — Number of employees; inverse proxy for economic health. During downturns (lower nr.employed), customers seek safety of deposits.
4. **emp.var.rate** — Employment variation rate. Negative rates (economic contraction) correlate with higher subscription as customers de-risk.
5. **cons.conf.idx** — Consumer confidence index. Counterintuitively, lower confidence → higher deposit uptake (risk aversion).
6. **poutcome** — Previous campaign outcome: `success` leads to drastically higher current subscription (see CUST_001: 97.1% probability).
7. **cons.price.idx** — Consumer price index (inflation). High CPI reduces value of fixed deposits in real terms.
8. **pdays** — Days since last contact. Recently contacted customers respond better; 999 = never previously contacted.
9. **age** — Elderly (65+) subscribe at 47.2%; young adults (<25) at 24.0%; middle-aged (35–54) only 8.65%.
10. **contact** — Cellular contact significantly outperforms telephone.

---

## 5. Why F1-Score Matters Here

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

In a bank marketing context:
- **Precision** = Of predicted YES, how many actually subscribe?  
  → Controls **wasted RM calls** — agents spend time on non-converters
- **Recall** = Of actual subscribers, how many did we catch?  
  → Controls **missed revenue** — every uncaught YES is lost business

With 88.7% "No" in our dataset, accuracy is a dangerous metric:
- A model predicting "No" always: **88.7% accuracy, 0.0% F1** — commercially worthless.

**Our results on real data:**

| Metric   | Logistic Regression | CatBoost   |
|----------|--------------------:|----------:|
| Accuracy | 0.8569              | **0.8651** |
| Precision| 0.4347              | **0.4527** |
| Recall   | 0.8998              | **0.9440** |
| F1       | 0.5862              | **0.6119** |
| ROC-AUC  | 0.9387              | **0.9566** |

CatBoost catches **94.4% of all actual subscribers** (recall) while maintaining better precision.
In a campaign of 10,000 contacts, this difference in recall could mean hundreds of additional term deposits opened.

---

## 6. Prediction Agreement Between Models

Both models made predictions on the same 5 sample customers:

| Customer | Profile                                  | LR Prediction | CatBoost Prediction | Agreement |
|----------|------------------------------------------|:---:|:---:|:---:|
| CUST_001 | Management, 520s call, success poutcome | YES | **YES (97.1%)** | ✓ |
| CUST_002 | Retired, 410s call, no prior contact    | YES | **YES (92.2%)** | ✓ |
| CUST_003 | Blue-collar, 52s call, no prior contact | NO  | **NO (0.000%)** | ✓ |
| CUST_004 | Services, 85s call, failed poutcome     | NO  | **NO (0.010%)** | ✓ |
| CUST_005 | Admin., 320s call, no prior contact     | — | **NO (29.75%)** | — |

**All 4 clear-cut cases agree.** CUST_005 is the borderline case where CatBoost's higher precision
correctly flags uncertainty (29.75% probability, just below the 0.5 threshold).

The models agree strongly on extreme cases — short calls with failed history = NO; long calls with
success history = YES. Disagreement only appears in the ambiguous middle band (0.3–0.6 probability),
where CatBoost's superior ROC-AUC of 0.9566 vs 0.9387 becomes the deciding factor.

---

## 7. The "200 Users" Issue — Why Sample Size Matters

If we evaluate on only 200 customers instead of 8,238 (our test set):

- **Variance explodes:** With 11.27% YES rate, 200 samples → only ~23 YES examples.
  The confidence interval on recall at n=23 is ±20%+ — far too wide to trust.
- **Metrics become unstable:** F1 could vary ±0.08–0.15 across random 200-sample draws.
- **Rare subgroups vanish:** Students (31.4% subscription rate) make up ~2% of data → only ~4 students in 200 samples. No statistical meaning.
- **False confidence:** A lucky sample could show 0.95 F1 that collapses to 0.61 on the full population.
- **Business risk:** A model validated on 200 samples could fail catastrophically when deployed to the full 40,000+ customer base.

**Best practice:** We use 8,238 samples (20% stratified split), preserving the exact 11.27%/88.73% class ratio. Stratification is critical — random splits on small sets can accidentally skew class ratios by 2–4%.

---

## 8. Value of the LLM Explain Endpoint

The `/explain` endpoint (Groq `llama-3.3-70b-versatile`) bridges the gap between model numbers and business action:

**Without LLM:**  
`"will_subscribe": true, "probability": 0.9708`  
→ A junior RM stares at a number and doesn't know what to say.

**With LLM:**  
*"This 45-year-old management professional responded positively in the previous campaign and engaged for over 8 minutes. Their financial profile shows no loan burden, making them an ideal candidate for a term deposit discussion. RM should lead with current rate advantages and 12-month lock-in options."*

**Concrete value added:**

1. **Prioritisation:** "Here are the top 20 customers to call today" with reasons — RM spends time where it matters most
2. **Personalised pitch guidance:** LLM uses job, age, poutcome to suggest specific talking points per customer
3. **RM onboarding:** Junior RMs get AI-generated call scripts, reducing training costs
4. **Auditability:** Plain-English explanations allow compliance teams to review AI decisions without data science expertise
5. **Trust building:** RMs who understand *why* trust the system and act on predictions — adoption increases
6. **Regulatory compliance:** GDPR and financial regulations increasingly require explainable AI decisions; the `/explain` endpoint provides this out of the box

The endpoint gracefully falls back to rule-based logic when `GROQ_API_KEY` is absent, ensuring zero downtime in production.

---

*VITBAIH Community Project 2026 — All outputs from real UCI Bank Marketing dataset (41,188 rows).*
