# Student Outcome Predictor (ML Assignment 2)

Live app: `<paste your Streamlit Cloud URL here>`
GitHub repository: https://github.com/Vishal-Bhushan-Research/MLAssignment2

## a. Problem statement

Higher-education institutions lose many students before graduation. If at-risk
students can be identified early, using information available at enrolment and
after the first two semesters, the institution can step in with tutoring,
counselling or financial support.

The task is a **3-class classification** problem: given a student's
demographic, socio-economic and academic-performance attributes, predict
whether the student will **Dropout**, remain **Enrolled** beyond the normal
course duration, or **Graduate**. Five classifiers are trained on the same
data, compared on six metrics, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, dataset id 697 (https://archive.ics.uci.edu/dataset/697) |
| Origin | Polytechnic Institute of Portalegre, Portugal (Realinho et al., 2022) |
| Instances | 4,424 students |
| Features | 36 (all numeric / integer-coded), no missing values |
| Target | `Status`: Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%) |

Feature groups:

* **Demographic**: marital status, nationality, gender, age at enrolment, displaced, international.
* **Socio-economic**: parents' qualifications and occupations, scholarship holder, debtor, tuition fees up to date, special educational needs.
* **Academic path at enrolment**: application mode and order, course, day/evening attendance, previous qualification and its grade, admission grade.
* **Academic performance**: curricular units credited / enrolled / evaluated / approved / grade / without evaluation, for the 1st and 2nd semester.
* **Macro-economic**: unemployment rate, inflation rate, GDP at the time of enrolment.

Preprocessing: categorical attributes are already integer-coded in the source
file. The data is split 80/20 with stratification on `Status`
(`random_state=2026`), giving 3,539 training rows and 885 test rows. The test
rows are saved as `test_data.csv`. `StandardScaler` is fitted inside the
pipeline (training data only) for Logistic Regression, KNN and SVM.

## c. Models used

All five models are scikit-learn `Pipeline` objects trained in
`train_models.py` and saved in `model/`:

| Model | Settings |
|---|---|
| Logistic Regression | StandardScaler + LogisticRegression(C=0.5, max_iter=3000) |
| Decision Tree | DecisionTreeClassifier(max_depth=7, min_samples_leaf=10) |
| KNN | StandardScaler + KNeighborsClassifier(n_neighbors=17, weights="distance") |
| Naive Bayes | GaussianNB() |
| Support Vector Machine | StandardScaler + SVC(kernel="rbf", C=1.0, probability=True) |

Because the target has three classes, Precision, Recall and F1 use the
**weighted** average and AUC is the **one-vs-rest weighted** ROC-AUC computed
from predicted probabilities. MCC is the multi-class Matthews correlation.

### Comparison table (hold-out test set, 885 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7593 | 0.8879 | 0.7416 | 0.7593 | 0.7423 | 0.5992 |
| Decision Tree | 0.7367 | 0.8656 | 0.7192 | 0.7367 | 0.7232 | 0.5616 |
| KNN | 0.7130 | 0.8497 | 0.7085 | 0.7130 | 0.6888 | 0.5224 |
| Naive Bayes | 0.6723 | 0.8116 | 0.6520 | 0.6723 | 0.6559 | 0.4501 |
| Support Vector Machine | 0.7605 | 0.8869 | 0.7483 | 0.7605 | 0.7442 | 0.6027 |

> If you re-run `train_models.py` on a different scikit-learn version the
> numbers can move slightly. Copy the table printed by the script.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | _TODO: write in your own words_ |
| Decision Tree | _TODO_ |
| KNN | _TODO_ |
| Naive Bayes | _TODO_ |
| Support Vector Machine | _TODO_ |
| Overall winner for this dataset | _TODO_ |

## Streamlit app

`app.py` loads the saved models (no retraining) and offers:

* CSV upload of test data (sidebar), plus a button to download the sample `test_data.csv`
* Model selection dropdown
* All six evaluation metrics for the chosen model
* Confusion matrix heatmap and classification report
* A "Compare all five" tab with a metric table and bar chart
* Download of the predictions as CSV

## Project structure

```
MLAssignment2/
├── app.py                 Streamlit app
├── train_models.py        trains the 5 models, writes model/ and test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv          20% hold-out split (36 features + Status)
├── data/
│   └── students_dropout.csv   original UCI file (';' separated)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── metadata.json      feature order, class labels, split info
    └── metrics.csv        metrics from training run
```

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py     # optional, models are already in model/
streamlit run app.py
```

## Reference

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5MC89
