
View in browser
Stop Using Brier Score Wrong
Brier score hides the most important distinction in probability evaluation. Here's the decomposition every ML practitioner should know.
VALERIY MANOKHIN
APR 4
∙
PAID

 




READ IN APP
 
My previous post post showed something uncomfortable. Platt scaling and isotonic regression — the two most common calibrators — degraded strong models across 30 datasets. Platt improved log-loss in only 49.8% of cases. A coin flip.

But here’s the question nobody asked: how were those models being evaluated in the first place?

If you’re using Brier score as a single number to judge probability quality, you’re conflating two properties that point in opposite directions. And that conflation is hiding bugs in your pipeline.

If you’re using Brier score as a single number to judge probability quality, you’re conflating two properties that point in opposite directions. And that conflation is hiding bugs in your pipeline.

What Brier Score Actually Measures

The Brier score is the mean squared error of your predicted probabilities:


Where p_i is the predicted probability and y_i is the true label (0 or 1). Lower is better. Simple.

Too simple. A model that predicts the base rate for every sample — say P = 0.3 for every observation when the class imbalance is 30/70 — gets a finite Brier score. It’s also completely useless. It can’t tell positive from negative.

A model that predicts P = 0.95 for true positives and P = 0.85 for true negatives is a strong discriminator. It separates the classes well. But it’s badly calibrated — the negatives should be much lower. Fix the calibration and you get a great model. The discrimination was already there.

Brier score mashes these two properties into one number. You can’t tell which one you have.

The Spiegelhalter Decomposition

David Spiegelhalter solved this in 1986. The Murphy decomposition — equivalent and more commonly cited — gives the same result. Here’s the breakdown:

Brier = Reliability - Resolution + Uncertainty

Three components. Each one tells you something different.

Reliability measures calibration. Group predictions into bins by predicted probability. For each bin, compare the predicted probability to the observed frequency. The gap between them — squared and weighted — is the reliability term. Lower is better. A perfectly calibrated model has reliability = 0.

Resolution measures discrimination. For each bin, compare the observed frequency to the overall base rate. A model that separates classes well will have bins with very different observed frequencies — some near 1.0, others near 0.0. High resolution means the model’s predictions carry real information about the outcome. Higher is better.

Uncertainty depends only on the data. It equals p_bar * (1 - p_bar) where p_bar is the overall positive rate. It’s constant across models for a given dataset. You can ignore it for model comparison.

The insight: a model can improve its Brier score by getting better at either calibration or discrimination. And a model with poor Brier can be an excellent discriminator that just needs calibration — or a well-calibrated model that can’t separate classes. You need the decomposition to know which.

The Decomposition Applied: 21 Classifiers, 30 Datasets

Our calibration study — arXiv:2601.19944 — benchmarked 21 classifiers across 30 binary tasks from TabArena-v0.1. The paper reports expected rank across 150 cross-validation folds. It does not compute the full Brier decomposition directly — that would require bin-level reliability and resolution estimates across every fold and dataset. But the paper gives us three metrics that serve as clean proxies for the three components.

Brier score rank captures overall probability quality — calibration and discrimination combined.

Spiegelhalter |Z|-statistic captures calibration. A model with low |Z| has predicted probabilities that match observed frequencies. This is the reliability term by another name.

AUC-ROC rank captures discrimination. A model with high AUC separates classes well — it pushes positive and negative predictions apart. This is what the resolution term measures.

Uncertainty depends only on the dataset. It’s constant across models. We can ignore it.

With these three proxies, the decomposition pattern becomes visible.


(Source: Figures 1–2 and Table 1 in arXiv:2601.19944)

The pattern is immediate.

CatBoost has a Brier rank of 3.99 — best among all GBDTs. Its |Z| is 1.88 — below the 1.96 significance threshold. Its AUC rank is 4.36. Both dimensions are strong. No single dimension is dragging it down.

XGBoost has a Brier rank of 10.87 — worst among the three GBDTs. But its AUC rank is 9.07. The gap between Brier and AUC is small. The problem is not discrimination — it’s calibration. The |Z| of 8.70 confirms it. XGBoost separates classes reasonably well but assigns probabilities that don’t match reality.

This is exactly what the decomposition predicts. XGBoost’s Brier score is bad because reliability is bad — not because resolution is bad. Fix the calibration and the Brier score drops. Friday’s post shows that Venn-Abers does exactly this — reducing XGBoost’s |Z| by 85.6% and improving log-loss by 12.6%.

Now look at Random Forest. Brier rank 8.55 — but AUC rank 7.79. The Brier rank is worse than the AUC rank suggests. RF’s vote-counting mechanism produces probabilities that cluster near the midrange — they’re wrong but not catastrophically wrong. Brier is forgiving of this. Log-loss is not — RF’s log-loss rank is 11.73, a 3-rank gap from Brier. Log-loss penalizes confident mistakes exponentially. RF makes plenty of those.

A direct Brier decomposition into bin-level reliability, resolution, and uncertainty — computed across all 150 folds — would sharpen these proxy-based conclusions. That analysis is feasible and would be a natural extension of the study. The proxies above give the directional story. The full decomposition would give the exact numbers.

How to Run the Decomposition Yourself

The algorithm is simple. Bin your predicted probabilities into 10 groups. For each bin — compute the average predicted probability and the observed frequency of positives. The squared gap between them, weighted by bin size, gives reliability. The squared gap between the observed frequency and the overall base rate, weighted by bin size, gives resolution. Uncertainty is just p_bar * (1 - p_bar) — a constant for any given dataset.

The Diagnostic Playbook

Once you have the decomposition, the decision tree is clear.

Low |Z|, high AUC: Well-calibrated AND discriminating. Ship it. In our 30-dataset study — CatBoost (|Z| 1.88, AUC rank 4.36), TabICL, and EBM land here.

High |Z|, high AUC: Bad calibration, good discrimination. This is XGBoost — |Z| of 8.70 but AUC rank 9.07. This is the best position to be in after “ship it” — because calibration is fixable. Apply Venn-Abers. You keep the discrimination and add guaranteed calibration.

High |Z|, low AUC: Bad calibration, bad discrimination. No post-hoc calibrator will save this. Retrain with a better model.

Low |Z|, low AUC: Well-calibrated but can’t separate classes. This is the “predicts the base rate” model. Technically calibrated. Practically useless.

The key insight: never optimize only for Brier score. Optimize for discrimination (AUC). Then fix calibration post-hoc if |Z| is above 1.96. The decomposition tells you which dimension needs work.

Why This Isn’t in Your Textbook

The Spiegelhalter decomposition has been in the statistics literature since 1986. The Murphy decomposition — mathematically equivalent — since 1973. They’re standard in weather forecasting, where probability calibration has been taken seriously for decades.

Machine learning largely ignored them. The field optimized for accuracy, then AUC, then log-loss. Brier score became popular as a “proper scoring rule” — which it is. But proper scoring rules still conflate calibration and discrimination. The decomposition separates them. Weather forecasters knew this 40 years ago.

Our study — 21 classifiers, 5 calibrators, 30 datasets — confirmed that this distinction matters. CatBoost and XGBoost make opposite trade-offs on the same metric. Without the decomposition, you’d pick the wrong one.

The Production Monitoring Case

The decomposition isn’t just for model selection. It’s essential for monitoring.

In production, Brier score can stay flat while both calibration and discrimination degrade — if they degrade in offsetting directions. You think the model is stable. It’s rotting.

Monitor reliability and resolution separately. Set alerts on both. When reliability drifts, recalibrate. When resolution drifts, retrain. Brier score alone can’t tell you which action to take.

The full monitoring stack:

Spiegelhalter Z-statistic for calibration drift detection (the Z-stat from the paper — measures whether reliability is significantly different from zero)
Resolution trend for discrimination monitoring
Venn-Abers for recalibration when reliability drifts
Paper: Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc Methods

If you want the full treatment — from Brier decomposition to conformal prediction sets to Venn-Abers calibration in production — it’s in my book: Applied Conformal Prediction
