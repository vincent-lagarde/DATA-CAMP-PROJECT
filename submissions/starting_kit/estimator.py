from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(lambda df: df["comment"].fillna(""),
                                  validate=False)

vectorizer = TfidfVectorizer(max_features=50,
                             ngram_range=(1, 2))
rf_clf = RandomForestClassifier(n_estimators=50)

pipe = Pipeline([
    ("text_isolator", transformer),
    ("tfidf", vectorizer),
    ("clf", rf_clf)
])


def get_estimator():
    return pipe
