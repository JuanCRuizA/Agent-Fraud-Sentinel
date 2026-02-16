"""
Generate Word Document: Model Training Notebook 03 - 4 Layers x 3 Perspectives
Agent Fraud Sentinel (BAFS) Project

Produces: docs/mt_03_analysis_matrix.docx
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures" / "model_training"
OUTPUT_PATH = BASE_DIR / "docs" / "mt_03_analysis_matrix.docx"

# ── Colors ─────────────────────────────────────────────────────────────
C_DARK_BLUE = "1B4F72"
C_MED_BLUE = "2980B9"
C_LIGHT_BLUE = "D6EAF8"
C_FRAUD_RED = "E74C3C"
C_LEGIT_GREEN = "2ECC71"
C_LIGHT_GRAY = "F2F3F4"
C_WHITE = "FFFFFF"
C_BLACK = "000000"
C_DARK_GRAY = "2C3E50"
C_INSIGHT_BG = "D6EAF8"
C_INSIGHT_BORDER = "2980B9"
C_BUSINESS_BG = "D5F5E3"
C_BUSINESS_BORDER = "27AE60"
C_CAUTION_BG = "FADBD8"
C_CAUTION_BORDER = "E74C3C"

LAYERS = ["WHAT did I do?", "WHY did I do it?", "HOW does it work?", "WHAT does the bank gain?"]

# ══════════════════════════════════════════════════════════════════════
#  CONTENT: ALL CHAPTER MATRICES
# ══════════════════════════════════════════════════════════════════════

CH0_MATRIX = {
    (0, 0): (
        "Trained two fraud detection models \u2014 Logistic Regression (baseline) and "
        "XGBoost (advanced) \u2014 on 354,324 transactions with 7 engineered features. "
        "Performed hyperparameter tuning (6 grid search combinations), cost-based "
        "threshold optimization, and designed a multi-threshold production strategy. "
        "Achieved 74.3% recall on the 118,108-row test set. Saved model artifacts "
        "(xgboost_final.pkl, scaler.pkl, threshold_config.pkl)."
    ),
    (0, 1): (
        "Built and compared two fraud detection systems. The simpler one (Logistic "
        "Regression) serves as a benchmark; the advanced one (XGBoost) is 33.8% better. "
        "Optimized the detection threshold based on real cost trade-offs: missing a fraud "
        "costs $75, a false alarm costs $10. Designed a three-tier strategy: auto-block "
        "obvious fraud, send borderline cases for human review, and auto-approve safe "
        "transactions."
    ),
    (0, 2): (
        "We taught two computer programs to spot fraud. The basic one is like a calculator "
        "\u2014 simple but limited. The advanced one is like a team of detectives working "
        "together. We found the best settings, figured out when to sound the alarm vs when "
        "to stay quiet, and created three action levels: block it, check it, or approve it."
    ),
    (1, 0): (
        "The 7 engineered features from Phase 2 need a classification model to convert "
        "signals into actionable fraud probabilities. Two models are compared to demonstrate "
        "that XGBoost's non-linear splits outperform linear models on this task. "
        "Cost-based threshold optimization translates the EDA's 7.5:1 cost ratio into "
        "a concrete decision boundary. The multi-threshold strategy addresses production "
        "operations (analyst workload management)."
    ),
    (1, 1): (
        "Having good fraud indicators (Phase 2) is not enough \u2014 the bank needs a "
        "system that combines them into a single fraud score for each transaction. "
        "Comparing two models proves the investment in a more complex system is justified. "
        "The cost analysis ensures the system balances fraud prevention with operational "
        "costs, and the three-tier strategy organizes the fraud team's workload."
    ),
    (1, 2): (
        "Having clues is one thing; putting them together to make a decision is another. "
        "We tried two approaches: a simple one and a smart one. The smart one is much "
        "better. Then we set the 'sensitivity dial' \u2014 how suspicious should the system "
        "be? Too suspicious and it bothers honest customers; not suspicious enough and "
        "it misses thieves."
    ),
    (2, 0): (
        "Logistic Regression with class_weight='balanced' and StandardScaler. XGBoost with "
        "scale_pos_weight=28.56 (inverse class frequency). Grid search over max_depth "
        "{4,6,8}, n_estimators {100,150,200}, learning_rate {0.05,0.1}. Best: depth=6, "
        "n_estimators=200, lr=0.05 (PR-AUC 0.1098). Threshold sweep 0.01\u20130.99 with "
        "cost function: total_cost = FN\u00d7$75 + FP\u00d7$10. Multi-threshold: auto-block "
        "\u22650.90, manual review \u22650.41, auto-approve <0.41."
    ),
    (2, 1): (
        "A simple statistical model and an advanced tree-based model are both trained on "
        "the same 7 features. The advanced model is tuned by testing 6 different "
        "configurations. The detection threshold is set by calculating the dollar cost "
        "of each possible cutoff point. The three-tier strategy splits transactions into "
        "automatic actions (block or approve) and human review."
    ),
    (2, 2): (
        "We trained two computer brains on the same clues. Then we tried 6 different "
        "'dial settings' on the smarter one to find the best combination. To set the "
        "alarm sensitivity, we calculated the cost of every possible setting and picked "
        "the one that balances catching thieves with not annoying honest customers."
    ),
    (3, 0): (
        "Production-ready model with documented performance: PR-AUC 0.1098, recall 74.3%, "
        "total cost $590K on 118K test transactions ($5.00/txn). Model artifacts are "
        "serialized and reproducible. The multi-threshold strategy enables differentiated "
        "operations: auto-block (0.02%), manual review (45.9%), auto-approve (54.1%)."
    ),
    (3, 1): (
        "The bank gets a fraud detection system that catches 3 out of 4 frauds while "
        "keeping costs manageable. The three-tier strategy means analysts only review "
        "borderline cases (46% of transactions), while clear-cut decisions are automated. "
        "All performance metrics are documented in dollar terms for executive reporting."
    ),
    (3, 2): (
        "The bank gets an alarm system that catches about 3 out of every 4 thieves. "
        "Obvious fraud is blocked instantly, suspicious cases go to a human checker, "
        "and clearly safe purchases go through without delay. The total cost of running "
        "this system is about $5 per transaction."
    ),
}

CH1_MATRIX = {
    (0, 0): (
        "Loaded 3 temporal-split CSVs from Phase 2: train (354,324 rows), val (118,108), "
        "test (118,108). Selected 7 features: txn_count_1hr, txn_count_24hr, "
        "amount_deviation, is_first_transaction, hour_of_day, is_weekend, TransactionAmt. "
        "Cleaned inf/NaN values (inf\u219210, -inf\u2192-10, NaN\u21920). Established cost "
        "assumptions: FN=$75, FP=$10, ratio 7.5:1."
    ),
    (0, 1): (
        "Loaded the three data files prepared in Phase 2 (training, practice, and final "
        "test sets). Selected the 7 fraud indicators to use as model inputs. Cleaned "
        "any problematic values. Documented the cost of errors: missing a fraud costs "
        "$75, a false alarm costs $10."
    ),
    (0, 2): (
        "We loaded the three data sets we prepared earlier and picked the 7 best clues "
        "to feed the computer. We fixed some messy data values, and reminded ourselves "
        "of the costs: missing a thief costs $75, bothering an honest customer costs $10."
    ),
    (1, 0): (
        "The 7 features were validated for fraud signal in Phase 2 (Tiers 1-4). "
        "amount_bin (Tier 4) is excluded to avoid redundancy with TransactionAmt. "
        "Infinity values arise from Z-score division by zero (single-transaction clients "
        "with std=0). Capping at \u00b110 preserves the extreme-deviation signal without "
        "causing numerical errors in StandardScaler or gradient computation."
    ),
    (1, 1): (
        "Only validated fraud indicators are used \u2014 not all 434 raw columns. This "
        "keeps the model focused and interpretable. The data cleaning step prevents "
        "mathematical errors that would crash the model. The cost assumptions from "
        "Phase 1 ensure every modeling decision is grounded in business reality."
    ),
    (1, 2): (
        "We only use the 7 clues we know work, not the hundreds of messy columns. "
        "Some values were broken (infinity or blanks), so we replaced them with "
        "reasonable numbers. And we kept our cost rules front and center so the "
        "computer learns what matters to the bank."
    ),
    (2, 0): (
        "Feature selection: ENGINEERED_FEATURES list + TransactionAmt. Data cleaning: "
        "df.replace([np.inf, -np.inf], [10, -10]).fillna(0). Assertions verify no "
        "inf/NaN remain. Cost constants: FN_COST=75.00, FP_COST=10.00, "
        "COST_RATIO=7.5. Scaler fit on training data only (leakage prevention)."
    ),
    (2, 1): (
        "Seven features are extracted from each dataset. Infinity values are replaced "
        "with 10 (extreme but finite), and blanks are filled with 0 (meaning 'no "
        "deviation'). The data is verified clean before modeling. Cost parameters "
        "are set as constants used throughout the notebook."
    ),
    (2, 2): (
        "We pulled out the 7 clues from each data file, replaced broken values with "
        "safe ones, double-checked everything was clean, and wrote down our cost rules "
        "so we could use them later."
    ),
    (3, 0): (
        "Clean, validated feature matrices (354K\u00d77, 118K\u00d77, 118K\u00d77) ready for "
        "modeling with no numerical issues. The inf\u219210 capping preserves extreme "
        "deviation signals (Z-score >10 occurs in <0.01% of cases) while ensuring "
        "numerical stability. Cost framework ready for threshold optimization."
    ),
    (3, 1): (
        "The bank starts with clean, reliable data and clear cost rules. No "
        "mathematical surprises will derail the modeling. The 7-feature design "
        "keeps the system fast enough for real-time scoring \u2014 critical for "
        "catching fraud at the point of transaction."
    ),
    (3, 2): (
        "The bank has clean, organized data ready to teach the computer. All the "
        "messy parts are fixed, the cost rules are clear, and the system is set "
        "up to learn from 354,000 past transactions."
    ),
}

CH2_MATRIX = {
    (0, 0): (
        "Trained Logistic Regression with class_weight='balanced', solver='lbfgs', "
        "max_iter=1000. Features standardized via StandardScaler (fit on train only). "
        "Validation PR-AUC: 0.0821. At threshold 0.5: precision 6.93%, recall 42.79%, "
        "F1 0.1193. Top coefficients: txn_count_1hr (0.259), is_first_transaction (-0.120)."
    ),
    (0, 1): (
        "Built a simple, interpretable model as the performance floor. This linear model "
        "catches 43% of fraud at the default threshold \u2014 decent but not enough for "
        "production. Its main value is as a benchmark: any advanced model must beat this "
        "to justify its complexity. Feature ranking confirms velocity is the top signal."
    ),
    (0, 2): (
        "We built a simple calculator-style model as our starting point. It catches "
        "about 4 out of 10 frauds. Not great, but it gives us a baseline to beat. "
        "It confirms that how fast someone shops (velocity) is the most important clue."
    ),
    (1, 0): (
        "Logistic Regression provides an interpretable benchmark. class_weight='balanced' "
        "adjusts the loss function to weight fraud samples ~28.6x more (inverse frequency). "
        "Without this, the model would predict all-legitimate (96.5% accuracy, 0% recall). "
        "PR-AUC is preferred over ROC-AUC because ROC-AUC inflates performance on "
        "imbalanced datasets by crediting true negatives."
    ),
    (1, 1): (
        "Every advanced model needs a simple benchmark to compare against. If the "
        "advanced model isn't significantly better, the bank should use the simpler one "
        "(easier to explain to regulators, cheaper to maintain). The 'balanced' setting "
        "prevents the model from taking the lazy route of saying 'nothing is fraud.'"
    ),
    (1, 2): (
        "We start simple to set a bar. If the simple model is good enough, why use "
        "something complicated? The simple model needs special instructions to pay "
        "attention to fraud \u2014 otherwise, since fraud is so rare, it would just "
        "say 'everything is fine' and be wrong 3.5% of the time."
    ),
    (2, 0): (
        "StandardScaler: z = (x - mean) / std, fit on X_train only. LogisticRegression("
        "class_weight='balanced'): internally scales positive class weight to n_samples / "
        "(n_classes * n_positive). predict_proba[:,1] gives fraud probability. "
        "precision_recall_curve + auc computes PR-AUC. Coefficients extracted from "
        "model.coef_[0] and sorted by absolute value."
    ),
    (2, 1): (
        "Features are first rescaled so they all have the same range (some are 0-23 "
        "for hours, others are 0-880 for velocity). The model learns a weight for "
        "each feature, then combines them into a fraud probability (0% to 100%). "
        "These weights reveal which features matter most."
    ),
    (2, 2): (
        "We first made all numbers the same scale (like converting inches and meters "
        "to the same unit). Then the model learned how important each clue is and "
        "combined them into a single 'fraud score' from 0 to 100%. The weights "
        "tell us which clues matter most."
    ),
    (3, 0): (
        "Establishes the performance floor: PR-AUC 0.0821. XGBoost must significantly "
        "exceed this to justify its complexity. The coefficients provide interpretable "
        "feature ranking consistent with Phase 2 signal analysis (velocity > first_txn "
        "> amount). The scaler is saved for deployment (ensures consistent feature "
        "transformation in production)."
    ),
    (3, 1): (
        "The bank now has a performance floor: any model chosen for production must beat "
        "PR-AUC 0.0821. The simple model's feature weights confirm the EDA findings "
        "(velocity is most important), which builds confidence in the feature engineering. "
        "If regulators require a simple model, this is ready to deploy."
    ),
    (3, 2): (
        "The bank has a starting point to compare against. The simple model confirms "
        "that our clues work (velocity is #1, just like we expected). If the bank "
        "wants the simplest possible system, this one is ready. But we can do better."
    ),
}

CH3_MATRIX = {
    (0, 0): (
        "Trained XGBoost with scale_pos_weight=28.56 (342,336/11,988), max_depth=6, "
        "n_estimators=100, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, "
        "eval_metric='aucpr'. Validation PR-AUC: 0.1093 (+33.2% over baseline). "
        "At threshold 0.5: recall 61.05%, precision 8.12%, F1 0.1433. "
        "Top features: TransactionAmt, txn_count_24hr."
    ),
    (0, 1): (
        "Built an advanced model (XGBoost) that outperforms the baseline by 33%. "
        "It catches 61% of fraud at the default threshold \u2014 up from 43% with "
        "the simple model. The model learns complex patterns like 'high velocity "
        "PLUS early morning PLUS unusual amount = very high fraud risk.'"
    ),
    (0, 2): (
        "We built a smarter model \u2014 think of it as a team of 100 small detectives, "
        "each asking different questions about the transaction. Together, they catch "
        "6 out of 10 frauds (up from 4 out of 10 with the simple model). The team "
        "learns that combinations of clues are more powerful than individual ones."
    ),
    (1, 0): (
        "XGBoost captures non-linear feature interactions that Logistic Regression cannot. "
        "For example, txn_count_1hr > 5 AND hour_of_day in [7,8,9] creates a compound "
        "risk signal not expressible as a linear combination. scale_pos_weight=28.56 "
        "reweights the gradient to handle the 3.38% fraud rate without oversampling "
        "(SMOTE), which can introduce synthetic artifacts."
    ),
    (1, 1): (
        "Fraud patterns are complex \u2014 a $200 purchase at 8 AM by someone who made "
        "5 purchases in the last hour is very different from the same amount at 2 PM "
        "by a regular customer. The simple model can't see these combinations; XGBoost "
        "can. The 28.56x weight tells XGBoost that each fraud case matters 28.56 times "
        "more than a normal case."
    ),
    (1, 2): (
        "The simple model looks at each clue separately, but fraud often involves "
        "combinations. A big purchase alone isn't suspicious, but a big purchase at "
        "7 AM after 10 other purchases in the last hour? Very suspicious. The smart "
        "model learns these combinations automatically."
    ),
    (2, 0): (
        "XGBoost builds sequential decision trees, each correcting the previous one's "
        "errors. scale_pos_weight modifies the gradient: loss_fraud = 28.56 * loss_legit. "
        "subsample=0.8 and colsample_bytree=0.8 provide regularization via row and "
        "column bagging. eval_metric='aucpr' monitors validation PR-AUC during training. "
        "No StandardScaler needed (tree-based models are scale-invariant)."
    ),
    (2, 1): (
        "The model builds 100 small decision trees, one at a time. Each new tree focuses "
        "on the mistakes the previous trees made. The fraud-weighting ensures the model "
        "pays 28.56 times more attention to fraud cases. Random subsampling of data and "
        "features prevents the model from memorizing the training data."
    ),
    (2, 2): (
        "Imagine 100 detectives taking turns. The first one does their best, then "
        "passes the hard cases to the second, who focuses on what the first missed. "
        "Each new detective learns from the mistakes of the ones before. Together, "
        "they form a much better team than any one detective alone."
    ),
    (3, 0): (
        "33.2% PR-AUC improvement over Logistic Regression (0.1093 vs 0.0821) justifies "
        "the model complexity. The feature importance ranking (TransactionAmt > "
        "txn_count_24hr > hour_of_day) differs from LR coefficients, suggesting "
        "non-linear effects dominate. This model becomes the candidate for threshold "
        "optimization and production deployment."
    ),
    (3, 1): (
        "The advanced model is significantly better (33% improvement), justifying its "
        "use despite being more complex. It catches 61% of fraud vs 43% for the "
        "simple model. The bank gets a stronger fraud detection engine that sees "
        "patterns the simple model misses."
    ),
    (3, 2): (
        "The smart model is much better \u2014 it catches 6 out of 10 frauds instead "
        "of 4. That's 50% more thieves caught. The extra complexity is worth it "
        "because it translates directly into less stolen money."
    ),
}

CH4_MATRIX = {
    (0, 0): (
        "Grid search over 6 parameter combinations: max_depth {4,6,8} \u00d7 "
        "n_estimators {100,150,200} \u00d7 learning_rate {0.05,0.1}. Best configuration: "
        "max_depth=6, n_estimators=200, learning_rate=0.05, PR-AUC=0.1098. "
        "Improvement over default XGBoost: +0.0005 PR-AUC (+0.5%). Final model "
        "retrained with best parameters."
    ),
    (0, 1): (
        "Fine-tuned the advanced model by testing 6 different configurations. The best "
        "one slightly outperforms the default (0.1098 vs 0.1093 PR-AUC). While the "
        "improvement is modest (+0.5%), it represents the best achievable performance "
        "within the sprint timeline. The final model is retrained with optimal settings."
    ),
    (0, 2): (
        "We tried 6 different 'dial settings' on the smart model to find the best "
        "combination. The improvement is small but real \u2014 like fine-tuning a "
        "radio to get the clearest signal. We picked the best settings and retrained "
        "the final model."
    ),
    (1, 0): (
        "Hyperparameter tuning searches for the model configuration that maximizes "
        "generalization (performance on unseen data). max_depth controls tree complexity "
        "(overfitting risk), learning_rate controls step size (convergence speed), "
        "n_estimators controls ensemble size (capacity). The small grid is appropriate "
        "for a 7-feature model where the parameter space is constrained."
    ),
    (1, 1): (
        "Every model has adjustable settings that affect how well it learns. Testing "
        "different combinations finds the sweet spot between learning too little "
        "(underfitting) and memorizing noise (overfitting). The validation set ensures "
        "the chosen settings work on new data, not just training data."
    ),
    (1, 2): (
        "The smart model has several 'knobs' we can turn. We tried turning them to "
        "different positions to find the best combination, like finding the right "
        "temperature for baking a cake \u2014 too low and it's raw, too high and "
        "it burns."
    ),
    (2, 0): (
        "Nested loop over param_grid. Each configuration trains a full XGBClassifier "
        "with eval_set=[(X_val, y_val)]. PR-AUC computed via precision_recall_curve + "
        "auc. Results sorted descending. Best params extracted via results_df.iloc[0]. "
        "Final model retrained with best params and full training set. "
        "Top 3: depth6/200/0.05 (0.1098), depth6/100/0.10 (0.1094), depth6/150/0.05 (0.1069)."
    ),
    (2, 1): (
        "Each of the 6 configurations is trained and tested on the validation set. "
        "The one with the highest PR-AUC score wins. The results show that moderate "
        "tree depth (6) with more trees (200) and a slower learning rate (0.05) "
        "produces the best model. The final model is retrained with these settings."
    ),
    (2, 2): (
        "We trained 6 different versions of the smart model and tested each one. "
        "The winner uses medium-sized trees, 200 of them, with a slow and steady "
        "learning approach. We picked the winner and trained the final model."
    ),
    (3, 0): (
        "Best achievable PR-AUC: 0.1098 (33.8% over LR baseline). The results show "
        "max_depth=6 consistently outperforms 4 and 8, confirming the model's sweet spot. "
        "The final model is saved with deterministic parameters (random_state=42) for "
        "full reproducibility. This is the model that enters threshold optimization."
    ),
    (3, 1): (
        "The bank gets the best possible model within the sprint timeline. The 33.8% "
        "improvement over the simple model is substantial. The tuning results are "
        "documented for audit trails \u2014 regulators can see exactly which "
        "configurations were tested and why the final one was chosen."
    ),
    (3, 2): (
        "We found the best possible version of the smart model. It's 33.8% better "
        "than the simple model. The bank can show inspectors exactly how we picked "
        "the settings, proving it wasn't guesswork."
    ),
}

CH5_MATRIX = {
    (0, 0): (
        "Compared Logistic Regression (PR-AUC 0.0821) vs XGBoost tuned (PR-AUC 0.1098) "
        "at threshold 0.5: LR recall 42.8%, XGB recall 60.9%. Plotted Precision-Recall "
        "curves. XGBoost selected for deployment (+33.8% PR-AUC improvement). "
        "Key: comparison performed at the same threshold for fairness."
    ),
    (0, 1): (
        "Side-by-side comparison of both models at the same threshold (0.5) ensures a "
        "fair evaluation. XGBoost wins on every metric: it catches 61% of fraud vs 43% "
        "for the simple model. The Precision-Recall curve shows XGBoost is consistently "
        "better across all possible operating points."
    ),
    (0, 2): (
        "We put both models to the test under identical conditions. The smart model "
        "catches 6 out of 10 frauds vs 4 out of 10 for the simple one. No matter "
        "how we adjust the sensitivity, the smart model is always better."
    ),
    (1, 0): (
        "Comparing models at different thresholds is a common pitfall \u2014 it conflates "
        "model quality with threshold selection. PR-AUC is threshold-independent, making "
        "it the correct metric for model selection. The PR curve comparison at the same "
        "threshold provides an additional visual fairness check. Only after model "
        "selection should threshold optimization begin."
    ),
    (1, 1): (
        "It would be unfair to compare the simple model at one sensitivity level and "
        "the advanced model at another. Like comparing two athletes, you must give them "
        "the same conditions. PR-AUC captures overall quality regardless of the "
        "sensitivity dial position. XGBoost wins clearly."
    ),
    (1, 2): (
        "It's like comparing two students on the same exam, not giving one an easier "
        "test. Both models get the same threshold, and the smart one scores higher "
        "on every measure. Only after picking the winner do we adjust its settings."
    ),
    (2, 0): (
        "precision_recall_curve(y_val, proba) computes precision and recall at all "
        "thresholds. auc(recall, precision) integrates the area. Fair comparison at "
        "threshold 0.5: (proba >= 0.5).astype(int). Metrics computed via sklearn: "
        "precision_score, recall_score, f1_score. PR curve plotted with baseline "
        "(y.mean()) for reference."
    ),
    (2, 1): (
        "Both models produce a fraud probability for each transaction. The PR curve "
        "traces precision vs recall at every possible cutoff. The area under this curve "
        "(PR-AUC) summarizes overall quality in a single number. A random model "
        "would achieve PR-AUC = 0.035 (the fraud rate), so both models add value."
    ),
    (2, 2): (
        "Both models give each transaction a score from 0 to 100%. We draw a line "
        "showing how many frauds they catch vs how many false alarms they make at "
        "every possible sensitivity level. The model with more area under the line wins."
    ),
    (3, 0): (
        "XGBoost selected with PR-AUC 0.1098 \u2014 13.4x better than a random classifier "
        "(0.0821/0.035 = 2.3x for LR; 0.1098/0.035 = 3.1x for XGB). The 33.8% "
        "relative improvement justifies the model complexity increase. This model "
        "now enters threshold optimization."
    ),
    (3, 1): (
        "The bank has a clear winner: XGBoost is 33.8% better and catches significantly "
        "more fraud. The selection process is documented and reproducible, meeting "
        "regulatory requirements for model selection justification (SR 11-7)."
    ),
    (3, 2): (
        "The bank has its champion model \u2014 the smart one, which is clearly better "
        "in every way. Now we need to decide how sensitive to make the alarm."
    ),
}

CH6_MATRIX = {
    (0, 0): (
        "Swept threshold 0.01\u20130.99 with cost function: total_cost = FN\u00d7$75 + FP\u00d7$10. "
        "Unconstrained optimum: threshold 0.740, cost $328K, recall 14.4% (unacceptable). "
        "Constrained (recall\u226575%): threshold 0.410, cost $598K, recall 76.0%. "
        "Trade-off: +$270K cost, +2,840 frauds caught. Cost per additional fraud: $95."
    ),
    (0, 1): (
        "Tested every possible threshold to find the cheapest one. The cheapest option "
        "($328K) only catches 14% of fraud \u2014 the bank misses 86% of thieves. "
        "By requiring at least 75% of fraud be caught, the cost rises to $598K, "
        "but 2,840 additional frauds are intercepted. This trade-off \u2014 $95 per "
        "extra fraud caught \u2014 is clearly worthwhile given the $75 median fraud loss."
    ),
    (0, 2): (
        "We tested every alarm setting from 'catch everything' to 'catch almost nothing.' "
        "The cheapest setting misses almost all fraud \u2014 terrible! So we added a rule: "
        "'you must catch at least 3 out of 4 frauds.' This costs more but prevents "
        "2,840 additional thefts."
    ),
    (1, 0): (
        "Pure cost minimization produces a degenerate solution: the threshold rises until "
        "the model barely flags anything, minimizing FP cost but accepting massive FN loss. "
        "The 14.4% recall at threshold 0.740 is operationally useless. The 75% recall "
        "constraint reflects regulatory expectations (banks must demonstrate good-faith "
        "fraud prevention) and customer trust requirements."
    ),
    (1, 1): (
        "Minimizing cost alone leads to a model that ignores most fraud \u2014 because "
        "false alarms are cheap and individual missed frauds are 'only' $75. But the "
        "bank can't afford to miss 86% of fraud: regulators, customers, and reputation "
        "all demand better. The 75% minimum recall is a business constraint that "
        "balances economics with responsibility."
    ),
    (1, 2): (
        "If we only care about cost, the system would barely raise any alarms \u2014 "
        "missing almost all thieves. That's like a security guard who never stops "
        "anyone because checking people is annoying. The bank needs the guard to "
        "stop at least 3 out of 4 suspicious people."
    ),
    (2, 0): (
        "For each threshold t in np.arange(0.01, 1.00, 0.01): y_pred = (proba >= t), "
        "compute FN = fraud missed, FP = legit flagged, cost = FN*75 + FP*10. "
        "Unconstrained: argmin(costs). Constrained: filter thresholds where "
        "recall >= 0.75, then argmin(filtered_costs). Trade-off: "
        "(constrained_cost - unconstrained_cost) / (constrained_TP - unconstrained_TP)."
    ),
    (2, 1): (
        "Every threshold from 1% to 99% is tested. For each, the system counts missed "
        "frauds and false alarms, then multiplies by their costs. The cheapest threshold "
        "is found. Then, adding the requirement to catch 75% of fraud, the cheapest "
        "threshold meeting that requirement is found."
    ),
    (2, 2): (
        "We tested 99 different sensitivity settings and calculated the cost of each. "
        "First we found the cheapest one (misses too much fraud). Then we found the "
        "cheapest one that still catches at least 3 out of 4 frauds."
    ),
    (3, 0): (
        "The constrained threshold (0.410) becomes the production manual-review boundary. "
        "The $270K cost increase prevents ~$213K in direct fraud losses (2,840 \u00d7 $75) "
        "plus indirect costs (customer churn, regulatory penalties). The cost-per-fraud "
        "metric ($95) provides a clear business justification for stakeholders."
    ),
    (3, 1): (
        "The bank gets a clear cost-benefit analysis: spending an extra $270K prevents "
        "2,840 additional frauds worth $213K in direct losses \u2014 plus avoiding "
        "customer anger, regulatory fines, and reputation damage that far exceed the "
        "direct cost. The 75% recall target is documented for audit compliance."
    ),
    (3, 2): (
        "The bank spends $270K more to catch 2,840 extra thieves. Those thefts would "
        "have cost at least $213K, plus the bank avoids angry customers and trouble "
        "with regulators. The extra cost is clearly worth it."
    ),
}

CH7_MATRIX = {
    (0, 0): (
        "Designed 3-tier production strategy: auto-block (\u22650.90, 0.02% of txns), "
        "manual review (0.41\u20130.90, 45.9%), auto-approve (<0.41, 54.1%). "
        "Test set: recall 74.3% (3,021/4,064 frauds), precision 5.6%, total cost "
        "$590K ($5.00/txn). Confusion matrix: TN=62,816, FP=51,228, FN=1,043, "
        "TP=3,021. Saved xgboost_final.pkl, scaler.pkl, threshold_config.pkl."
    ),
    (0, 1): (
        "Created an operational system with three action levels: (1) obvious fraud is "
        "blocked instantly \u2014 no human needed, (2) borderline cases go to analysts "
        "for review, (3) clearly safe transactions pass through. The final test shows "
        "the system catches 74.3% of fraud on data it has never seen before."
    ),
    (0, 2): (
        "The system has three levels: 'Stop!' for obvious fraud (blocked automatically), "
        "'Check this' for suspicious cases (a person reviews it), and 'Go ahead' for "
        "clearly safe purchases. Testing on new data shows it catches about 3 out of "
        "every 4 thieves."
    ),
    (1, 0): (
        "A single threshold creates a binary decision, but fraud operations require "
        "graduated responses. Auto-block (\u22650.90) handles the 0.02% of transactions "
        "where the model has near-certainty, reducing latency and analyst workload. "
        "Manual review (0.41\u20130.90) escalates uncertain cases to human judgment. "
        "Auto-approve (<0.41) frees 54% of transactions from any friction."
    ),
    (1, 1): (
        "Not all fraud decisions are equal. Some are clear-cut (score 0.95 = definitely "
        "fraud), some are borderline (score 0.60 = maybe), and some are clearly fine "
        "(score 0.10 = definitely legitimate). A single yes/no threshold wastes human "
        "analysts on clear-cut cases. Three tiers match the response to the confidence level."
    ),
    (1, 2): (
        "Imagine a traffic light: red means 'stop' (obvious fraud), yellow means "
        "'slow down and check' (suspicious), and green means 'go ahead' (safe). "
        "Without this system, every slightly suspicious transaction would need a "
        "human checker, overwhelming the team."
    ),
    (2, 0): (
        "Score segmentation: auto_block = (proba >= 0.90), manual_review = "
        "(proba >= 0.41) & (proba < 0.90), auto_approve = (proba < 0.41). "
        "Cost: auto-block FP \u00d7 $5 + manual FP \u00d7 $10 + missed fraud \u00d7 $75. "
        "Test evaluation: apply thresholds to final_xgb_proba_test. "
        "Artifacts: joblib.dump(model, scaler, threshold_config)."
    ),
    (2, 1): (
        "Each transaction's fraud score determines its action: above 90% gets blocked "
        "automatically, 41-90% goes to an analyst, below 41% passes through. The test "
        "set (never seen during training) provides the final unbiased performance estimate. "
        "All model files are saved for deployment."
    ),
    (2, 2): (
        "The computer gives each purchase a suspicion score. High scores are blocked "
        "automatically, medium scores go to a human checker, and low scores pass "
        "through. We tested this on data the computer never saw before to make sure "
        "it really works. Then we saved everything for real-world use."
    ),
    (3, 0): (
        "Test set recall 74.3% (near the 75% target) with precision 5.6% and $590K "
        "total cost. The auto-block tier catches 4 frauds with 19 FPs (21% precision "
        "\u2014 much higher than overall). Auto-approve correctly handles 62,816 legit "
        "transactions. Model artifacts are serialized for Phase 4 (SHAP) and Phase 5 "
        "(Streamlit dashboard)."
    ),
    (3, 1): (
        "The bank gets a battle-tested system: 74.3% fraud detection on completely "
        "new data. The three-tier approach means 54% of transactions need zero human "
        "involvement. Obvious fraud is stopped instantly. Analysts focus only on "
        "borderline cases (46% of transactions). Everything is saved and ready for "
        "the explainability review (Phase 4) and the dashboard (Phase 5)."
    ),
    (3, 2): (
        "The final test shows the system works: it catches 3 out of 4 thieves on "
        "data it has never seen. More than half of all transactions go through "
        "without anyone needing to check them. The system is saved and ready to "
        "be explained (Phase 4) and put into a control panel (Phase 5)."
    ),
}

ALL_CHAPTERS = [
    {
        "number": 0,
        "title": "Executive Overview",
        "subtitle": "Full-Project Summary",
        "narrative": (
            "This notebook takes the 7 engineered features from Phase 2 and trains them into "
            "a production-ready fraud detection model. Two models are compared: Logistic "
            "Regression (interpretable baseline) and XGBoost (advanced). XGBoost wins by "
            "33.8% (PR-AUC 0.1098 vs 0.0821). Cost-based threshold optimization using the "
            "EDA's $75/$10 cost assumptions produces a three-tier production strategy: auto-block "
            "(score \u22650.90), manual review (0.41\u20130.90), and auto-approve (<0.41). "
            "On the 118,108-transaction test set, the system achieves 74.3% recall at a cost "
            "of $5.00 per transaction."
        ),
        "matrix": CH0_MATRIX,
        "figures": [],
        "callouts": [
            ("insight",
             "The notebook follows a disciplined two-step process: first select the best model "
             "(using PR-AUC at a fixed threshold for fair comparison), then optimize the "
             "threshold (using cost analysis with business constraints). Mixing these steps "
             "is a common pitfall."),
        ],
    },
    {
        "number": 1,
        "title": "Setup & Data Preparation",
        "subtitle": "Notebook Section 1: Loading, Feature Selection, Data Cleaning, Cost Assumptions",
        "narrative": (
            "The pipeline begins by loading the three temporal-split CSV files from Phase 2 "
            "and selecting the 7 validated features. Infinity values in amount_deviation "
            "(from Z-score division by zero) are capped at \u00b110. NaN values are filled "
            "with 0. The cost framework from Phase 1 is carried forward: FN=$75, FP=$10, "
            "ratio 7.5:1."
        ),
        "matrix": CH1_MATRIX,
        "figures": [],
        "callouts": [
            ("business",
             "The 7.5:1 cost ratio is the foundation of every modeling decision. It means "
             "the bank should tolerate up to 7.5 false alarms for every fraud it catches \u2014 "
             "because missing a fraud is 7.5 times more expensive than investigating a false alarm."),
        ],
    },
    {
        "number": 2,
        "title": "Baseline Model: Logistic Regression",
        "subtitle": "Notebook Section 2: Training, Evaluation, Feature Coefficients",
        "narrative": (
            "Logistic Regression with balanced class weights establishes the performance floor. "
            "Features are standardized (mean=0, std=1) via StandardScaler fit on training data "
            "only. The baseline achieves PR-AUC 0.0821 with 42.8% recall at threshold 0.5. "
            "Feature coefficients confirm velocity (txn_count_1hr: 0.259) as the strongest "
            "linear signal, consistent with Phase 2 findings."
        ),
        "matrix": CH2_MATRIX,
        "figures": [],
        "callouts": [
            ("insight",
             "Logistic Regression's coefficients provide the only truly interpretable feature "
             "ranking in this pipeline. txn_count_1hr (0.259) dominates, confirming that "
             "transaction velocity is the most important linear fraud signal."),
        ],
    },
    {
        "number": 3,
        "title": "Advanced Model: XGBoost",
        "subtitle": "Notebook Section 3: Training with Class Imbalance Handling",
        "narrative": (
            "XGBoost with scale_pos_weight=28.56 captures non-linear feature interactions. "
            "The initial model (default hyperparameters) achieves PR-AUC 0.1093, a 33.2% "
            "improvement over the baseline. At threshold 0.5, recall jumps from 42.8% to "
            "61.1%. Feature importance shifts: TransactionAmt and txn_count_24hr become "
            "dominant, suggesting non-linear amount patterns that Logistic Regression missed."
        ),
        "matrix": CH3_MATRIX,
        "figures": [
            ("xgb_feature_importance.png",
             "Figure 1: XGBoost Feature Importance (Gain). TransactionAmt and txn_count_24hr "
             "dominate, revealing non-linear amount and velocity patterns missed by Logistic Regression."),
        ],
        "callouts": [
            ("business",
             "XGBoost catches 18% more fraud than the simple model (61% vs 43%) because it "
             "detects complex patterns: 'high velocity + unusual amount + early morning' is "
             "far more suspicious than any single indicator alone."),
        ],
    },
    {
        "number": 4,
        "title": "Hyperparameter Tuning",
        "subtitle": "Notebook Section 4: Grid Search (6 Configurations)",
        "narrative": (
            "A targeted grid search tests 6 parameter combinations, varying max_depth (4, 6, 8), "
            "n_estimators (100, 150, 200), and learning_rate (0.05, 0.1). The best configuration "
            "(depth=6, 200 trees, lr=0.05) achieves PR-AUC 0.1098 \u2014 a modest +0.5% over "
            "the default. The consistent dominance of max_depth=6 confirms the model's "
            "complexity sweet spot."
        ),
        "matrix": CH4_MATRIX,
        "figures": [],
        "callouts": [
            ("insight",
             "The small tuning improvement (+0.5%) indicates that the default XGBoost parameters "
             "were already near-optimal. With only 7 features, the model's performance is more "
             "constrained by feature quality than by hyperparameters \u2014 validating the Phase 2 "
             "feature engineering approach."),
        ],
    },
    {
        "number": 5,
        "title": "Model Selection & Threshold Optimization",
        "subtitle": "Notebook Section 5 (Part 1): Fair Comparison, PR Curves, Cost Sweep",
        "narrative": (
            "Model selection is performed first (XGBoost wins with PR-AUC 0.1098), then "
            "threshold optimization begins. A cost sweep from 0.01 to 0.99 reveals a "
            "tension: pure cost minimization (threshold 0.740, $328K) catches only 14.4% "
            "of fraud. Constraining recall \u226575% yields threshold 0.410 at $598K \u2014 "
            "an additional $270K that catches 2,840 more frauds."
        ),
        "matrix": CH5_MATRIX,
        "figures": [
            ("pr_curve_comparison.png",
             "Figure 2: Precision-Recall Curve \u2014 Model Comparison. XGBoost Tuned (PR-AUC 0.1098) "
             "consistently outperforms Logistic Regression (PR-AUC 0.0821) across all recall levels. "
             "The red dashed line represents the no-model baseline (3.9% fraud rate)."),
        ],
        "callouts": [
            ("caution",
             "Pure cost minimization (14.4% recall) is a degenerate solution: the model "
             "essentially ignores fraud because individual misses ($75) are cheap relative "
             "to review costs ($10 each for 51K false alarms). The 75% recall constraint "
             "is a business guardrail against this pathological optimization."),
        ],
    },
    {
        "number": 6,
        "title": "Cost-Based Threshold Optimization",
        "subtitle": "Notebook Section 5 (Part 2): Unconstrained vs Constrained, Trade-Off Analysis",
        "narrative": (
            "The threshold optimization reveals the core business trade-off. Unconstrained "
            "optimization (threshold 0.740) minimizes cost at $328K but catches only 14.4% "
            "of fraud \u2014 unacceptable. Constraining recall \u226575% raises the threshold "
            "to 0.410 with $598K cost but catches 76% of fraud. The extra $270K prevents "
            "2,840 additional frauds worth $213K in direct losses plus reputation and "
            "regulatory costs."
        ),
        "matrix": CH6_MATRIX,
        "figures": [
            ("cost_vs_threshold.png",
             "Figure 3: Cost vs Threshold Optimization. The U-shaped curve shows total cost "
             "across all thresholds. Left side: low threshold = many false alarms (high FP cost). "
             "Right side: high threshold = many missed frauds (high FN cost). Red dashed line marks "
             "the unconstrained optimum at 0.740."),
        ],
        "callouts": [
            ("business",
             "The trade-off in dollars: spending $270K more catches 2,840 additional frauds "
             "worth at least $213K in direct losses. When factoring in customer churn, "
             "regulatory penalties, and reputation damage, the 75% recall strategy is "
             "overwhelmingly justified."),
        ],
    },
    {
        "number": 7,
        "title": "Production Strategy & Test Set Evaluation",
        "subtitle": "Notebook Section 5 (Part 3): Multi-Threshold, Confusion Matrix, Model Export",
        "narrative": (
            "The final production strategy uses three tiers: auto-block (\u22650.90), manual "
            "review (0.41\u20130.90), and auto-approve (<0.41). On the 118,108-transaction "
            "test set, the system achieves 74.3% recall with $590K total cost ($5.00/txn). "
            "54% of transactions are auto-approved with zero friction. Model artifacts "
            "(xgboost_final.pkl, scaler.pkl, threshold_config.pkl) are saved for Phase 4 "
            "explainability and Phase 5 dashboard deployment."
        ),
        "matrix": CH7_MATRIX,
        "figures": [
            ("confusion_matrix_absolute.png",
             "Figure 4: Confusion Matrix \u2014 Absolute Numbers (Test Set). Shows the count of "
             "true negatives (62,816), false positives (51,228), false negatives (1,043), and "
             "true positives (3,021) at the production threshold of 0.41."),
            ("confusion_matrix_percentages.png",
             "Figure 5: Confusion Matrix \u2014 Percentages (Test Set). The same confusion matrix "
             "expressed as percentages of total transactions. 53.18% are correctly auto-approved, "
             "while 2.56% of fraud is correctly caught."),
        ],
        "callouts": [
            ("business",
             "The production system auto-approves 54% of transactions instantly and only "
             "sends 46% to human review. Obvious fraud (score \u22650.90) is blocked without "
             "analyst intervention. This three-tier design balances fraud prevention with "
             "operational efficiency and customer experience."),
        ],
    },
]

SUMMARY_TABLE_DATA = [
    ("Models Compared", "Logistic Regression vs XGBoost"),
    ("Winner", "XGBoost (PR-AUC 0.1098, +33.8%)"),
    ("Features Used", "7 (from Phase 2 tiers 1-3 + TransactionAmt)"),
    ("Training Set", "354,324 rows (60%), fraud rate 3.38%"),
    ("Validation Set", "118,108 rows (20%), fraud rate 3.90%"),
    ("Test Set", "118,108 rows (20%), fraud rate 3.44%"),
    ("Baseline PR-AUC (LR)", "0.0821"),
    ("Final PR-AUC (XGBoost)", "0.1098"),
    ("Best Hyperparameters", "depth=6, n_estimators=200, lr=0.05"),
    ("Unconstrained Threshold", "0.740 (cost $328K, recall 14.4%)"),
    ("Constrained Threshold (75%)", "0.410 (cost $598K, recall 76.0%)"),
    ("Auto-Block Threshold", "\u2265 0.90"),
    ("Manual Review Range", "0.41 \u2013 0.90"),
    ("Test Recall", "74.3% (3,021 of 4,064 frauds)"),
    ("Test Precision", "5.6%"),
    ("Test Total Cost", "$590,410 ($5.00/txn)"),
    ("Cost Ratio (FN:FP)", "7.5:1 ($75 vs $10)"),
    ("Model Artifacts", "xgboost_final.pkl, scaler.pkl, threshold_config.pkl"),
]

GLOSSARY = [
    ("Auto-Approve", "Transactions with fraud scores below the review threshold (<0.41) are automatically approved with no human review."),
    ("Auto-Block", "Transactions with very high fraud scores (\u22650.90) are automatically blocked without waiting for human review."),
    ("Class Weight (Balanced)", "A Logistic Regression setting that automatically weights fraud samples higher to compensate for their rarity (3.5%)."),
    ("Confusion Matrix", "A 2x2 table showing True Negatives, False Positives, False Negatives, and True Positives."),
    ("Constrained Optimization", "Finding the best solution (lowest cost) while meeting a requirement (e.g., catch at least 75% of fraud)."),
    ("Cost Sweep", "Testing every possible threshold and calculating the total cost at each one to find the optimal operating point."),
    ("F1-Score", "The harmonic mean of precision and recall. A balanced metric, but less useful than PR-AUC for imbalanced data."),
    ("Grid Search", "Testing multiple combinations of model settings to find the best configuration."),
    ("Hyperparameters", "Model settings that control how the model learns (e.g., tree depth, number of trees, learning speed)."),
    ("Manual Review", "Transactions with borderline fraud scores (0.41-0.90) are sent to human analysts for investigation."),
    ("PR-AUC", "Precision-Recall Area Under Curve. The primary metric for comparing models on imbalanced fraud data."),
    ("Precision", "Of all transactions flagged as fraud, what percentage are actually fraud? Low precision = many false alarms."),
    ("Recall (Sensitivity)", "Of all actual frauds, what percentage does the model catch? 74.3% recall = catches 3 of 4 frauds."),
    ("Scale Pos Weight", "An XGBoost parameter (28.56) that tells the model each fraud case is worth 28.56 legitimate cases."),
    ("StandardScaler", "A preprocessing step that rescales features to mean=0 and standard deviation=1, required for Logistic Regression."),
    ("Threshold", "The fraud probability cutoff above which a transaction is flagged. Lower = more sensitive; higher = more selective."),
    ("XGBoost", "Extreme Gradient Boosting. Builds many small decision trees sequentially, each correcting the previous one's errors."),
]


# ══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def set_cell_shading(cell, color_hex):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color_hex)
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=50, bottom=50, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in [("top", top), ("bottom", bottom), ("start", left), ("end", right)]:
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        tcMar.append(el)
    tcPr.append(tcMar)


def set_paragraph_spacing(paragraph, before=0, after=0, line=240):
    pPr = paragraph._p.get_or_add_pPr()
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), str(before))
    spacing.set(qn("w:after"), str(after))
    spacing.set(qn("w:line"), str(line))
    spacing.set(qn("w:lineRule"), "auto")
    pPr.append(spacing)


def add_formatted_text(cell, text, font_name="Calibri", font_size=10,
                       bold=False, color_hex=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_spacing(p, before=20, after=20, line=240)
    run = p.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if color_hex:
        run.font.color.rgb = RGBColor.from_string(color_hex)


def add_callout_box(doc, text, box_type="insight"):
    colors = {
        "insight": (C_INSIGHT_BG, C_INSIGHT_BORDER, "Key Insight"),
        "business": (C_BUSINESS_BG, C_BUSINESS_BORDER, "Business Impact"),
        "caution": (C_CAUTION_BG, C_CAUTION_BORDER, "Caution"),
    }
    bg_color, border_color, label = colors.get(box_type, colors["insight"])

    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=120, after=120, line=264)

    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), bg_color)
    shd.set(qn("w:val"), "clear")
    pPr.append(shd)

    pBdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:color"), border_color)
    left.set(qn("w:space"), "4")
    pBdr.append(left)
    pPr.append(pBdr)

    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "360")
    ind.set(qn("w:right"), "360")
    pPr.append(ind)

    label_run = p.add_run(f"{label}: ")
    label_run.font.name = "Calibri"
    label_run.font.size = Pt(10)
    label_run.font.bold = True
    label_run.font.color.rgb = RGBColor.from_string(border_color)

    content_run = p.add_run(text)
    content_run.font.name = "Calibri"
    content_run.font.size = Pt(10)
    content_run.font.italic = True
    content_run.font.color.rgb = RGBColor.from_string(C_DARK_GRAY)


def add_figure(doc, image_filename, caption_text):
    img_path = FIGURES_DIR / image_filename
    if img_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(img_path), width=Inches(5.5))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_paragraph_spacing(cap, before=40, after=200, line=240)
        cap_run = cap.add_run(caption_text)
        cap_run.font.name = "Calibri"
        cap_run.font.size = Pt(9)
        cap_run.font.italic = True
        cap_run.font.color.rgb = RGBColor.from_string("666666")
    else:
        p = doc.add_paragraph(
            f"[Figure not available: {image_filename} \u2014 run notebook to generate]"
        )
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.italic = True
        p.runs[0].font.color.rgb = RGBColor.from_string("999999")


def add_matrix_table(doc, matrix_data):
    table = doc.add_table(rows=5, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), "9360")
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)

    borders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "BFBFBF")
        border.set(qn("w:space"), "0")
        borders.append(border)
    tblPr.append(borders)

    headers = [
        "Layer", "Technical\n(BDS Colleague)",
        "Business\n(Manager / Regulator)", "Simple\n(Grandmother)"
    ]
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        set_cell_shading(cell, C_DARK_BLUE)
        set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
        add_formatted_text(cell, header, font_size=10, bold=True, color_hex=C_WHITE)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, layer in enumerate(LAYERS):
        row_idx = i + 1
        layer_cell = table.cell(row_idx, 0)
        set_cell_shading(layer_cell, C_MED_BLUE)
        set_cell_margins(layer_cell, top=60, bottom=60, left=100, right=100)
        add_formatted_text(layer_cell, layer, font_size=9, bold=True, color_hex=C_WHITE)
        layer_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for j in range(3):
            cell = table.cell(row_idx, j + 1)
            bg = C_WHITE if i % 2 == 0 else C_LIGHT_GRAY
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            text = matrix_data.get((i, j), "")
            add_formatted_text(cell, text, font_size=9)

    doc.add_paragraph("")
    return table


def add_section_heading(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor.from_string(C_DARK_BLUE)
    return heading


def add_page_break(doc):
    doc.add_page_break()


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def add_cover_page(doc):
    for _ in range(6):
        p = doc.add_paragraph()
        set_paragraph_spacing(p, before=0, after=0)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Agent Fraud Sentinel")
    run.font.name = "Calibri"
    run.font.size = Pt(32)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(C_DARK_BLUE)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Model Training Deep Dive: 4 Layers \u00d7 3 Perspectives")
    run.font.name = "Calibri"
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor.from_string(C_MED_BLUE)

    sep = doc.add_paragraph()
    sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sep.add_run("\u2500" * 50)
    run.font.color.rgb = RGBColor.from_string(C_MED_BLUE)
    run.font.size = Pt(12)

    ref = doc.add_paragraph()
    ref.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ref.add_run("Notebook: 03_model_training.ipynb")
    run.font.name = "Consolas"
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor.from_string(C_DARK_GRAY)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run(
        "IEEE-CIS Fraud Detection Dataset\n"
        "Logistic Regression vs XGBoost | Cost-Based Threshold Optimization\n"
        "Multi-Threshold Production Strategy | 74.3% Recall"
    )
    run.font.name = "Calibri"
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor.from_string(C_DARK_GRAY)

    for _ in range(3):
        doc.add_paragraph()

    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run("BAFS Project \u2014 February 2026")
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor.from_string("888888")

    add_page_break(doc)


def add_toc_placeholder(doc):
    add_section_heading(doc, "Table of Contents", level=1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run()
    run.font.name = "Calibri"
    run.font.size = Pt(11)

    fldChar_begin = OxmlElement("w:fldChar")
    fldChar_begin.set(qn("w:fldCharType"), "begin")
    run._r.append(fldChar_begin)
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = ' TOC \\o "1-3" \\h \\z \\u '
    run._r.append(instrText)
    fldChar_separate = OxmlElement("w:fldChar")
    fldChar_separate.set(qn("w:fldCharType"), "separate")
    run._r.append(fldChar_separate)

    placeholder_run = p.add_run(
        "(Right-click here and select 'Update Field' to generate Table of Contents)"
    )
    placeholder_run.font.name = "Calibri"
    placeholder_run.font.size = Pt(10)
    placeholder_run.font.italic = True
    placeholder_run.font.color.rgb = RGBColor.from_string("999999")
    fldChar_end = OxmlElement("w:fldChar")
    fldChar_end.set(qn("w:fldCharType"), "end")
    placeholder_run._r.append(fldChar_end)

    add_page_break(doc)


def add_framework_explanation(doc):
    add_section_heading(doc, "The 4-Layer \u00d7 3-Perspective Framework", level=2)
    p = doc.add_paragraph()
    run = p.add_run(
        "This document examines each section of the Model Training notebook through "
        "two dimensions: four analytical layers and three audience perspectives. This "
        "framework ensures complete understanding \u2014 from raw technical detail to "
        "business impact \u2014 accessible to any reader regardless of their background."
    )
    run.font.name = "Calibri"
    run.font.size = Pt(11)

    add_section_heading(doc, "The Four Layers", level=3)
    for title_text, desc in [
        ("Layer 1 \u2014 WHAT did I do?",
         "Describes the concrete actions: models trained, thresholds tested, strategies designed."),
        ("Layer 2 \u2014 WHY did I do it?",
         "Explains the motivation: why two models, why cost-based optimization, why three tiers."),
        ("Layer 3 \u2014 HOW does it work?",
         "Details the mechanics: algorithms, cost functions, threshold selection, model saving."),
        ("Layer 4 \u2014 WHAT does the bank gain?",
         "Translates results into value: fraud caught, cost savings, operational efficiency."),
    ]:
        p = doc.add_paragraph()
        bold_run = p.add_run(title_text + "  ")
        bold_run.font.name = "Calibri"
        bold_run.font.size = Pt(11)
        bold_run.font.bold = True
        bold_run.font.color.rgb = RGBColor.from_string(C_DARK_BLUE)
        desc_run = p.add_run(desc)
        desc_run.font.name = "Calibri"
        desc_run.font.size = Pt(11)

    add_section_heading(doc, "The Three Perspectives", level=3)
    for title_text, desc in [
        ("Technical (BDS Colleague)",
         "Uses ML terminology, references hyperparameters, metrics, and code patterns."),
        ("Business (Manager / Regulator)",
         "Focuses on cost trade-offs, compliance, and operational strategy. No coding assumed."),
        ("Simple (Grandmother)",
         "Uses everyday analogies and plain language. No technical background assumed."),
    ]:
        p = doc.add_paragraph()
        bold_run = p.add_run(title_text + "  ")
        bold_run.font.name = "Calibri"
        bold_run.font.size = Pt(11)
        bold_run.font.bold = True
        bold_run.font.color.rgb = RGBColor.from_string(C_MED_BLUE)
        desc_run = p.add_run(desc)
        desc_run.font.name = "Calibri"
        desc_run.font.size = Pt(11)


def add_production_strategy_table(doc):
    """Add the multi-threshold strategy table unique to this notebook."""
    add_section_heading(doc, "Production Strategy: Three-Tier Decision Framework", level=2)

    data = [
        ("Auto-Block", "\u2265 0.90", "Instant block (automated)",
         "$5.00", "0.02% of transactions"),
        ("Manual Review", "0.41 \u2013 0.90", "Analyst investigation",
         "$10.00", "45.9% of transactions"),
        ("Auto-Approve", "< 0.41", "Approved (no action)",
         "$0.00", "54.1% of transactions"),
    ]

    table = doc.add_table(rows=len(data) + 1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    borders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "BFBFBF")
        border.set(qn("w:space"), "0")
        borders.append(border)
    tblPr.append(borders)

    for j, header in enumerate(["Tier", "Score Range", "Action", "Cost/Txn", "Volume"]):
        cell = table.cell(0, j)
        set_cell_shading(cell, C_DARK_BLUE)
        set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
        add_formatted_text(cell, header, font_size=10, bold=True, color_hex=C_WHITE)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    tier_colors = [C_FRAUD_RED, "F39C12", C_LEGIT_GREEN]  # red, orange, green
    for i, (tier, score, action, cost, volume) in enumerate(data):
        bg = C_WHITE if i % 2 == 0 else C_LIGHT_GRAY
        for j, text in enumerate([tier, score, action, cost, volume]):
            cell = table.cell(i + 1, j)
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=30, bottom=30, left=80, right=80)
            add_formatted_text(cell, text, font_size=10, bold=(j == 0))
            if j > 0:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")


def add_chapter(doc, chapter_data):
    num = chapter_data["number"]
    title = chapter_data["title"]

    add_page_break(doc)
    add_section_heading(doc, f"Chapter {num}: {title}", level=1)

    if chapter_data.get("subtitle"):
        p = doc.add_paragraph()
        run = p.add_run(chapter_data["subtitle"])
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        run.font.italic = True
        run.font.color.rgb = RGBColor.from_string(C_MED_BLUE)

    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=120, after=200, line=276)
    run = p.add_run(chapter_data["narrative"])
    run.font.name = "Calibri"
    run.font.size = Pt(11)

    add_section_heading(doc, "Analysis Matrix", level=2)
    add_matrix_table(doc, chapter_data["matrix"])

    for box_type, text in chapter_data.get("callouts", []):
        add_callout_box(doc, text, box_type)

    for fig_filename, caption in chapter_data.get("figures", []):
        add_figure(doc, fig_filename, caption)


def add_summary_statistics_table(doc):
    add_section_heading(doc, "Model Training Summary", level=2)

    table = doc.add_table(rows=len(SUMMARY_TABLE_DATA) + 1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    borders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "BFBFBF")
        border.set(qn("w:space"), "0")
        borders.append(border)
    tblPr.append(borders)

    for j, header in enumerate(["Metric", "Value"]):
        cell = table.cell(0, j)
        set_cell_shading(cell, C_DARK_BLUE)
        set_cell_margins(cell, top=40, bottom=40, left=100, right=100)
        add_formatted_text(cell, header, font_size=10, bold=True, color_hex=C_WHITE)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, (metric, value) in enumerate(SUMMARY_TABLE_DATA):
        bg = C_WHITE if i % 2 == 0 else C_LIGHT_GRAY
        for j, text in enumerate([metric, value]):
            cell = table.cell(i + 1, j)
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=30, bottom=30, left=100, right=100)
            add_formatted_text(cell, text, font_size=10, bold=(j == 0))
            if j == 1:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


GALLERY_FIGURES = [
    ("xgb_feature_importance.png",
     "Figure 1: XGBoost Feature Importance (Gain). Shows relative importance of each "
     "of the 7 engineered features. TransactionAmt and txn_count_24hr dominate, "
     "revealing non-linear patterns that Logistic Regression cannot capture."),
    ("pr_curve_comparison.png",
     "Figure 2: Precision-Recall Curve \u2014 Model Comparison. XGBoost Tuned "
     "(PR-AUC 0.1098) outperforms Logistic Regression (PR-AUC 0.0821) across all "
     "recall levels. The red dashed baseline represents the no-model fraud rate (3.9%)."),
    ("cost_vs_threshold.png",
     "Figure 3: Cost vs Threshold Optimization. The U-shaped total cost curve balances "
     "FP cost (left, low thresholds) against FN cost (right, high thresholds). The "
     "unconstrained optimum at 0.740 minimizes cost but catches only 14.4% of fraud."),
    ("confusion_matrix_absolute.png",
     "Figure 4: Confusion Matrix \u2014 Absolute Numbers (Test Set, 118,108 transactions). "
     "At the production threshold of 0.41: TN=62,816, FP=51,228, FN=1,043, TP=3,021. "
     "The system catches 74.3% of fraud at 5.6% precision."),
    ("confusion_matrix_percentages.png",
     "Figure 5: Confusion Matrix \u2014 Percentages (Test Set). Same data expressed as "
     "percentages: 53.18% correctly auto-approved, 43.37% flagged for review, "
     "0.88% fraud missed, 2.56% fraud correctly caught."),
]


def add_appendix_gallery(doc):
    add_page_break(doc)
    add_section_heading(doc, "Appendix A: Visualization Gallery", level=1)

    p = doc.add_paragraph()
    run = p.add_run(
        "All model training visualizations from notebook 03, presented with detailed captions."
    )
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    set_paragraph_spacing(p, after=200)

    for fig_filename, caption in GALLERY_FIGURES:
        add_figure(doc, fig_filename, caption)


def add_appendix_glossary(doc):
    add_page_break(doc)
    add_section_heading(doc, "Appendix B: Glossary", level=1)

    p = doc.add_paragraph()
    run = p.add_run(
        "Key terms used throughout this document, defined for non-technical readers."
    )
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    set_paragraph_spacing(p, after=200)

    table = doc.add_table(rows=len(GLOSSARY) + 1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    borders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "BFBFBF")
        border.set(qn("w:space"), "0")
        borders.append(border)
    tblPr.append(borders)

    for j, header in enumerate(["Term", "Definition"]):
        cell = table.cell(0, j)
        set_cell_shading(cell, C_DARK_BLUE)
        set_cell_margins(cell, top=40, bottom=40, left=100, right=100)
        add_formatted_text(cell, header, font_size=10, bold=True, color_hex=C_WHITE)

    for i, (term, definition) in enumerate(GLOSSARY):
        bg = C_WHITE if i % 2 == 0 else C_LIGHT_GRAY
        cell_term = table.cell(i + 1, 0)
        set_cell_shading(cell_term, bg)
        set_cell_margins(cell_term, top=30, bottom=30, left=100, right=100)
        add_formatted_text(cell_term, term, font_size=10, bold=True)

        cell_def = table.cell(i + 1, 1)
        set_cell_shading(cell_def, bg)
        set_cell_margins(cell_def, top=30, bottom=30, left=100, right=100)
        add_formatted_text(cell_def, definition, font_size=10)


def setup_header_footer(doc):
    for section in doc.sections:
        header = section.header
        header.is_linked_to_previous = False
        h_para = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        h_para.text = ""
        run_left = h_para.add_run("Agent Fraud Sentinel \u2014 Model Training")
        run_left.font.name = "Calibri"
        run_left.font.size = Pt(8)
        run_left.font.color.rgb = RGBColor.from_string("999999")
        h_para.add_run("\t\t")
        run_right = h_para.add_run("BAFS")
        run_right.font.name = "Calibri"
        run_right.font.size = Pt(8)
        run_right.font.bold = True
        run_right.font.color.rgb = RGBColor.from_string(C_DARK_BLUE)

        footer = section.footer
        footer.is_linked_to_previous = False
        f_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        f_para.text = ""
        f_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_page = f_para.add_run("Page ")
        run_page.font.name = "Calibri"
        run_page.font.size = Pt(8)
        run_page.font.color.rgb = RGBColor.from_string("999999")
        fldChar1 = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")
        run_page._r.append(fldChar1)
        instrText = OxmlElement("w:instrText")
        instrText.set(qn("xml:space"), "preserve")
        instrText.text = " PAGE "
        run_page._r.append(instrText)
        fldChar2 = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "separate")
        run_page._r.append(fldChar2)
        fldChar3 = OxmlElement("w:fldChar")
        fldChar3.set(qn("w:fldCharType"), "end")
        run_page._r.append(fldChar3)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Generating Model Training Analysis Matrix document...")

    doc = Document()

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for i in range(1, 4):
        h_style = doc.styles[f"Heading {i}"]
        h_style.font.name = "Calibri"
        h_style.font.color.rgb = RGBColor.from_string(C_DARK_BLUE)

    # ── Build Document ─────────────────────────────────────────────
    add_cover_page(doc)
    add_toc_placeholder(doc)

    # Chapter 0
    ch0 = ALL_CHAPTERS[0]
    add_section_heading(doc, f"Chapter 0: {ch0['title']}", level=1)
    p = doc.add_paragraph()
    run = p.add_run(ch0["subtitle"])
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.italic = True
    run.font.color.rgb = RGBColor.from_string(C_MED_BLUE)

    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=120, after=200, line=276)
    run = p.add_run(ch0["narrative"])
    run.font.name = "Calibri"
    run.font.size = Pt(11)

    add_framework_explanation(doc)
    add_production_strategy_table(doc)

    add_section_heading(doc, "Full-Project Analysis Matrix", level=2)
    add_matrix_table(doc, ch0["matrix"])

    for box_type, text in ch0.get("callouts", []):
        add_callout_box(doc, text, box_type)

    # Chapters 1-7
    for chapter_data in ALL_CHAPTERS[1:]:
        add_chapter(doc, chapter_data)

    # Summary
    add_summary_statistics_table(doc)

    # Appendix A: Visualization Gallery
    add_appendix_gallery(doc)

    # Appendix B: Glossary
    add_appendix_glossary(doc)

    # Header/Footer
    setup_header_footer(doc)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_PATH))
    print(f"Document saved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
