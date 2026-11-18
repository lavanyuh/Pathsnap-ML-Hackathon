# PathSnap: Distributed Map-Matching & Route Reconstruction

Distributed machine learning system for vehicle route reconstruction from noisy GPS trajectories: **LightGBM Candidate Classifier + Topological HMM / Viterbi Sequence Decoder**, served via **FastAPI** on Laptop 1 with an interactive dashboard on Laptop 2.

Developed for **UE24CS352A - Machine Learning Hackathon (Department of CSE, PES University)**.

---

## 🚀 Quick Start & Two-Laptop Deployment

### 1. Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/lavanyuh/Pathsnap-ML-Hackathon.git
cd Pathsnap-ML-Hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Two-Laptop Evaluation Architecture

```
┌──────────────────────────────────────────────┐        Wi-Fi / LAN         ┌──────────────────────────────────────────────┐
│                  LAPTOP 1                    │     HTTP REST / JSON       │                  LAPTOP 2                    │
│             (Backend Server)                 │ ◄────────────────────────► │             (Evaluator UI)                   │
│                                              │                            │                                              │
│ • FastAPI Server (0.0.0.0:8000)              │   1. Select Trip ID        │ • Clean Web Dashboard (ui/index.html)        │
│ • Trained LightGBM Model Artifact            │   2. Process Prediction    │ • Lists 10 Reserved Evaluation Trips         │
│ • HMM / Viterbi Decoder                      │   3. Return Metrics & Geo  │ • Live Metrics Cards (Accuracy, Continuity)  │
│ • Server-Side Ground Truth (.route)          │                            │ • Interactive Leaflet Map (GPS vs Predicted) │
│ • Formatted Terminal Audit Logs              │                            │                                              │
└──────────────────────────────────────────────┘                            └──────────────────────────────────────────────┘
```

#### On Laptop 1 (Model Server & API):
Run the server launcher:
```bash
python run_server.py
```
* The API runs on `http://0.0.0.0:8000`.
* Finds your LAN IP address (e.g. `ipconfig getifaddr en0` on macOS or `hostname -I` on Linux/Windows).
* Shows formatted live audit logs for every evaluator inference request in the terminal.

#### On Laptop 2 (Evaluator Client):
Open your web browser and navigate directly to:
```
http://<LAPTOP_1_IP>:8000/
```
*(Alternatively, open `ui/index.html` in Chrome and enter `http://<LAPTOP_1_IP>:8000` into the API Endpoint field).*

---

## 🏆 Model Selection & Bayesian Optimization

### 1. 5-Model Family Benchmark
Per hackathon guidelines, five distinct machine learning algorithm families were benchmarked on the candidate road segment classification task:

| Algorithm Family | Model | Accuracy | Precision | Recall | Macro F1 | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|---|
| **Linear** | Logistic Regression (L2) | 92.50% | 0.5527 | 0.9392 | 0.8266 | 0.9835 | 0.38s |
| **Probabilistic** | Gaussian Naive Bayes | 91.15% | 0.5087 | 0.9164 | 0.8018 | 0.9645 | 0.05s |
| **Tree Ensemble** | Random Forest (100 trees) | 96.75% | 0.7533 | 0.9582 | 0.9127 | 0.9938 | 8.74s |
| **Boosting (1)** | HistGradientBoosting | 97.84% | 0.8875 | 0.8745 | 0.9345 | 0.9943 | 2.67s |
| **Boosting (2)** | **LightGBM (Tuned - Winner)** | **97.94%** | **0.8918** | **0.8816** | **0.9377** | **0.9947** | **2.76s** |

> **Selection Rationale:** **LightGBM** was selected as our final production model because it achieved the **highest Macro F1 Score (0.9377)** and highest **ROC-AUC (0.9947)**, while offering ultra-low inference latency (**0.004 ms per candidate**).

---

### 2. Bayesian Hyperparameter Optimization (Optuna)
Bayesian hyperparameter optimization was executed over 15 trials maximizing Macro F1 score on validation splits (`scripts/08_train_lgbm_bayesopt.py`):

* **Search Space & Tuned Hyperparameters:**
  * `learning_rate`: `0.04587`
  * `num_leaves`: `70`
  * `max_depth`: `9`
  * `min_child_samples`: `116`
  * `subsample`: `0.7992`
  * `colsample_bytree`: `0.9957`
  * `reg_alpha`: `0.0177`
  * `reg_lambda`: `0.0654`
  * `scale_pos_weight`: `1.1194`
* **Artifact Output:** Saved to `artifacts/classifier.joblib` and trial history saved to `results/models/lgbm/optuna_trials.csv`.

---

## 🗺️ Sequential Route Reconstruction: HMM & Viterbi

While the classifier predicts the emission probability $P(\text{Arc} \mid \text{Point})$, GPS noise and multi-path errors can cause isolated erroneous jumps. We formulate an HMM where:
1. **Hidden States:** Candidate road segments within a 50m spatial query radius.
2. **Emission Probabilities:** Calibrated probabilities output by the tuned LightGBM model.
3. **Transition Probabilities:** Exponential penalty based on discrepancy between network distance and Euclidean GPS displacement:
   $$P(s_t \mid s_{t-1}) \propto \exp\left(-\frac{|\Delta d_{\text{network}} - \Delta d_{\text{GPS}}|}{\beta}\right)$$
   * **Intra-Arc Continuity:** For consecutive points on the same segment, $\Delta d_{\text{network}} = |t_2 - t_1| \cdot L$, absorbing GPS jitter without penalizing slow moving or stopped vehicles.
   * **Inter-Arc Adjacency:** Transition across topological road connections $B(a) = A(b)$ uses remaining segment length $(1 - t_1) \cdot L_a + t_2 \cdot L_b$.
   * **Sequence Segmentation:** Gaps exceeding 5 seconds segment the Viterbi decoding into independent sequences.

---

## 📊 Phase 2: Evaluation on the 10 Reserved Trips (`evaluator_test`)

Batch evaluation across all 10 reserved evaluator test trips (`scripts/10_evaluate_all_evaluator_trips.py`):

| Trip ID | GPS Points | Candidate Rows | HMM Accuracy | Classifier Accuracy | HMM Gain | Route Continuity | Breaks | Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000015` | 6,854 | 55,290 | **85.41%** | 68.71% | **+16.70%** | 94.81% | 14 | 2,069 ms |
| `00000017` | 1,346 | 33,110 | **89.67%** | 93.76% | -4.09% | 92.99% | 45 | 1,032 ms |
| `00000023` | 458 | 4,929 | **94.32%** | 94.54% | -0.22% | 94.74% | 3 | 177 ms |
| `00000030` | 654 | 15,805 | **78.59%** | 79.66% | -1.07% | 93.80% | 17 | 456 ms |
| `00000049` | 2,642 | 14,287 | **99.43%** | 99.02% | +0.42% | 99.51% | 2 | 809 ms |
| `00000059` | 1,031 | 19,850 | **93.99%** | 96.02% | -2.04% | 95.53% | 17 | 568 ms |
| `00000067` | 916 | 6,348 | **95.09%** | 94.76% | +0.33% | **100.00%** | 0 | 233 ms |
| `00000073` | 1,144 | 11,312 | **97.73%** | 97.38% | +0.35% | 99.58% | 1 | 420 ms |
| `00000087` | 1,399 | 13,636 | **98.86%** | 94.78% | **+4.07%** | 99.10% | 4 | 673 ms |
| `00000094` | 3,230 | 39,000 | **94.46%** | 92.88% | **+1.58%** | 98.22% | 11 | 1,171 ms |
| **AVERAGE** | **1,967** | **21,357** | **92.76%** | **91.15%** | **+1.61%** | **96.83%** | **11.4** | **761 ms** |

---

## 📁 Repository Structure

```
├── artifacts/
│   ├── classifier.joblib             # Final tuned LightGBM model artifact
│   └── preprocess.joblib             # Train-only standardizer/imputer
├── data/                             # Features & point tables (train, student_test)
├── results/
│   ├── eda/                          # Exploratory plots and dataset summaries
│   ├── features/                     # Candidate feature distribution plots
│   └── models/
│       ├── lgbm/                     # Optuna trial logs, confusion matrix, feature importance
│       ├── model_comparison_summary.csv # 5-model comparative metrics table
│       ├── model_comparison_chart.png   # 5-model comparison bar chart
│       └── evaluator_test_summary.csv   # Phase 2 10-trip evaluation summary
├── scripts/
│   ├── 01_inventory.py               # Dataset inventory and EDA
│   ├── 02_folium_map.py              # Exploratory Folium visualization
│   ├── 04_derive_labels.py           # Point label derivation from reference route
│   ├── 05_build_features.py          # Spatial candidate generation and feature extraction
│   ├── 08_train_lgbm_bayesopt.py     # LightGBM training with Optuna Bayesian optimization
│   ├── 09_compare_five_models.py     # 5-model family benchmarking script
│   └── 10_evaluate_all_evaluator_trips.py # Phase 2 batch evaluator script
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI backend implementation
│   ├── data/
│   │   ├── geo.py                    # Geodesic distances and local projection
│   │   ├── labels.py                 # Ground truth reference label extraction
│   │   ├── loader.py                 # Raw dataset loader
│   │   └── network.py                # STRtree spatial index & road network topology
│   ├── eval/
│   │   └── route_metrics.py          # Continuity and sequence F1 evaluation
│   ├── features/
│   │   ├── build.py                  # Candidate generation & feature engineering
│   │   └── columns.py                # Single source of truth for feature columns
│   └── models/
│       ├── hmm.py                    # Viterbi decoding algorithm
│       └── pipeline.py               # Complete end-to-end MapMatchingPipeline
├── ui/
│   └── index.html                    # Evaluator-facing web interface (Leaflet.js + Tailwind)
├── run_server.py                     # Laptop 1 server entry point
└── requirements.txt                  # Python dependencies
```

---

## 🎓 Viva & Demonstration Defense Points

1. **Why LightGBM?**
   * Tree boosting natively captures non-linear relationships and interactions among spatial, kinematic, and contextual features (e.g., `dist_rel_min`, `heading_cos`, `win3_mean_dist`).
   * Handles class imbalance (~1:9 positive-to-negative candidate ratio) effectively, giving the highest Macro F1 score (0.9377).
   * Blazing-fast inference (~4 µs per candidate) enables real-time map matching on thousands of points in sub-second time.

2. **Why HMM / Viterbi?**
   * Independent candidate classification ignores route connectivity; a vehicle cannot teleport across disconnected roads.
   * Viterbi guarantees finding the globally optimal, topologically continuous sequence across all observations.
   * Notice Trip `00000015`: HMM improves pointwise accuracy from **68.71% to 85.41% (+16.70% net gain)** by eliminating jitter jumps.

3. **Zero Data Leakage:**
   * Ground truth `.route` reference files are strictly reserved on the backend (Laptop 1) for post-inference accuracy calculation.
   * Features (`build_trip_features`) are calculated exclusively from candidate road geometry and GPS track kinematics without labels.
