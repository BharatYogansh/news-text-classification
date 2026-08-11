"""
Shared config: category list, artifact paths. One place to change the
category set or add a 4th model without hunting through every file.
"""
CATEGORIES = [
    "comp.graphics",
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.misc",
    "sci.med",
]

MODEL_NAMES = ["logistic_regression", "naive_bayes", "linear_svm"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Multinomial Naive Bayes",
    "linear_svm": "Linear SVM",
}

VECTORIZER_PATH = "vectorizer.pkl"
MODEL_PATH_TMPL = "model_{name}.pkl"
RESULTS_PATH = "results.json"
TEST_DATA_PATH = "test_predictions.csv"
