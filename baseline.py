# %%
import pandas as pd

df = pd.read_csv("data/manager_salary.csv")
df.head()

# %%
df.info()

# %% [markdown]
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
df_vectorizer = vectorizer.fit_transform(df)

# %%
df_vectorizer.head()

# %%
for name, transformer,  columns in vectorizer.transformers_:
    print(f"{name}: {transformer.__class__.__name__} applied on columns {columns}")

# %% [markdown]
# During the parsing, the `TableVectorizer` did not parse well the income column.
# The second issue is about encoding free text.
