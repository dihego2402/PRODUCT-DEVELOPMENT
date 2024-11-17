from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def train_model(model_name, X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB()
    }
    if model_name not in models:
        raise ValueError(f"Modelo {model_name} no es válido. Elige entre {list(models.keys())}")
    
    model = models[model_name]
    model.fit(X_train, y_train)
    return model
