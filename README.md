# Gender Classification

A binary classification project predicting gender from facial/physical measurements, comparing Logistic Regression, KNN, and a Decision Tree. Built in August 2024.

## Dataset

- `gender_classification.csv`, 5,001 rows originally
- Features: `long_hair`, `forehead_width_cm`, `forehead_height_cm`, `nose_wide`, `nose_long`, `lips_thin`, `distance_nose_to_lip_long` (all binary except the two width/height measurements)
- Target: `gender` (Male/Female)
- 1,768 duplicate rows (about 35% of the dataset) are dropped during preprocessing, leaving 3,233 rows
- After dedup, the class balance shifts slightly: 1,783 Male (55.1%) vs 1,450 Female (44.9%), compared to a near-even 2,500/2,501 split before dedup

## Approach

1. Drop duplicate rows
2. Encode gender with `LabelEncoder` (Female = 0, Male = 1)
3. Scale features with `MinMaxScaler`
4. 80/20 train/test split (`random_state=42`)
5. Train and compare Logistic Regression, KNN (k=3), and a Decision Tree

## Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 95.1% | **96.0%** |
| KNN (k=3) | 97.0% | 94.4% |
| Decision Tree | 99.8% | 94.4% |

Logistic Regression, the simplest of the three, generalized best. The Decision Tree shows the largest train/test gap (99.8% vs 94.4%), the same overfitting pattern seen in a couple of the earlier projects in this series.

**Logistic Regression classification report:**

| | Precision | Recall | F1 |
|---|---|---|---|
| Female | 0.955 | 0.961 | 0.958 |
| Male | 0.964 | 0.959 | 0.962 |

Balanced performance across both classes, no class is being favored at the other's expense.

## Which features actually mattered

Looking at the Logistic Regression coefficients on the scaled features:

| Feature | Coefficient magnitude |
|---|---|
| nose_wide | 3.60 |
| lips_thin | 3.34 |
| distance_nose_to_lip_long | 3.28 |
| nose_long | 3.27 |
| forehead_width_cm | 2.18 |
| forehead_height_cm | 1.81 |
| long_hair | 0.07 |

`long_hair`, despite being the single most stereotype-associated feature in the dataset, carries almost no weight in the model (0.07, next to nothing compared to the others). The facial measurement features do essentially all of the work.

## Corrections to the original README

The original README described the model as using "features such as age and height." Neither age nor height exist anywhere in this dataset. The actual features are the seven facial/physical measurements listed above. This rewrite reflects the real columns.

There's also a small copy-paste bug in the original script: the print statement for the Decision Tree's accuracy prints the KNN variable (`knn_test_acc`) instead of the tree's own variable. It's cosmetic only, the correct value (`model_test_acc`) is what actually gets used in the final comparison table and the model-selection logic, so it doesn't affect any of the results above.

## Tech Stack

Python, Pandas, NumPy, Seaborn, Matplotlib, scikit-learn (`LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `MinMaxScaler`, `LabelEncoder`)

## How to Run

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
python gender_classification.py
```

## What I Learned

Two things stood out on revisiting this. First, over a third of the raw dataset was duplicate rows, a reminder to always check `duplicated().sum()` before trusting any dataset's size or class balance. Second, looking at the model's actual coefficients instead of just its accuracy showed that the feature I'd have guessed mattered most (hair length) barely contributed anything, while the facial measurements did nearly all the work. Accuracy tells you a model works; the coefficients tell you why.
