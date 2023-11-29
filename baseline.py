# %%
import pandas as pd

df = pd.read_csv("data/manager_salary.csv")
df.head()

# %%
df.info()

# %%
import numpy as np
from sklearn.model_selection import train_test_split


def initialization(df):
    df = df[
        [
            "Timestamp",
            "How old are you?",
            "What is your annual salary? (You'll indicate the currency in a later question. If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.)",
            "How much additional monetary compensation do you get, if any (for example, bonuses or overtime in an average year)? Please only include monetary compensation here, not the value of benefits.",
            "Please indicate the currency",
            "What is your gender?",
        ]
    ]
    target_column_name = "What is your gender?"
    # remove rows with missing target values
    df = df.dropna(subset=[target_column_name])
    # remove rows where the target is not Man or Woman
    df = df[np.isin(df["What is your gender?"], ["Man", "Woman"])]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_train = df_train[target_column_name]
    y_test = df_test[target_column_name]
    X_train = df_train.drop(columns=[target_column_name])
    X_test = df_test.drop(columns=[target_column_name])
    X_train_shape_original = X_train.shape
    X_test_shape_original = X_test.shape
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_shape_original,
        X_test_shape_original,
    )


(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_shape_original,
    X_test_shape_original,
) = initialization(df)

# %%
X, y = pd.concat([X_train, X_test], axis=0), pd.concat([y_train, y_test], axis=0)


# %%
# We would expect to parse the data in the following manner:
#  0   Timestamp - 27996 non-null  datetime
#  1   How old are you? - 27996 non-null  category
#  2   What industry do you work in? - 27924 non-null  category
#  3   Job title - 27996 non-null  category
#  4   If your job title needs additional context, please clarify here - 7243 non-null   object
#  5   What is your annual salary? - 27996 non-null  numeric
#  6   How much additional monetary compensation do you get, if any
#      (for example, bonuses or overtime in an average year)? Please only include monetary
#      compensation here, not the value of benefits. - 20722 non-null  numeric
#  7   Please indicate the currency - 27996 non-null  category
#  8   If "Other," please indicate the currency here: - 197 non-null    category
#  9   If your income needs additional context, please provide it here: - 3034 non-null   object
#  10  What country do you work in? - 27996 non-null  category
#  11  If you're in the U.S., what state do you work in? - 22998 non-null  category
#  12  What city do you work in? - 27916 non-null  category
#  13  How many years of professional work experience do you have overall? - 27996 non-null  category
#  14  How many years of professional work experience do you have in your field? - 27996 non-null  category
#  15  What is your highest level of education completed? - 27780 non-null  category
#  16  What is your gender? - 27829 non-null  category
#  17  What is your race? (Choose all that apply.) - 27824 non-null  category

# %%
from skrub import TableVectorizer

vectorizer = TableVectorizer().set_output(transform="pandas")
X_train_vectorize = vectorizer.fit_transform(X_train)

# %%
X_train_vectorize.head()

# %%
for name, transformer, columns in vectorizer.transformers_:
    print(f"{name}: {transformer.__class__.__name__} applied on columns {columns}")

# %% [markdown]
# During the parsing, the `TableVectorizer` did not parse well the income column.
# The second issue is about encoding free text.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

pipeline = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingClassifier(random_state=42),
)
pipeline.fit(X_train, y_train)
y_pred_gap_women = pipeline.predict_proba(X_test)[:, 1]

# %%
from sklearn.metrics import roc_auc_score

print(f"Test score: {roc_auc_score(y_test, y_pred_gap_women)}")

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    pipeline, X, y, cv=5, scoring="roc_auc", return_train_score=True
)
cv_results = pd.DataFrame(cv_results)

# %%
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %%
from skrub import MinHashEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

pipeline = make_pipeline(
    TableVectorizer(high_cardinality_transformer=MinHashEncoder()),
    HistGradientBoostingClassifier(random_state=42),
)
pipeline.fit(X_train, y_train)
y_pred_minhash_women = pipeline.predict_proba(X_test)[:, 1]

# %%

print(f"Test score: {roc_auc_score(y_test, y_pred_minhash_women)}")

# %%
# %%
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

_, ax = plt.subplots(figsize=(6, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_pred_gap_women,
    name="Model using Gap Encoder",
    pos_label="Woman",
    plot_chance_level=False,
    ax=ax,
)
RocCurveDisplay.from_predictions(
    y_test,
    y_pred_minhash_women,
    name="Model using MinHash Encoder",
    pos_label="Woman",
    plot_chance_level=True,
    ax=ax,
)

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    pipeline, X, y, cv=5, scoring="roc_auc", return_train_score=True
)
cv_results = pd.DataFrame(cv_results)

# %%
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %%
