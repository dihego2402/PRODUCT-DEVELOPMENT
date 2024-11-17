from sklearn.model_selection import RandomizedSearchCV

def optimize_hyperparameters(model, param_distributions, X_train, y_train, n_trials):
    search = RandomizedSearchCV(model, param_distributions, n_iter=n_trials, cv=3, scoring='f1_macro', n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_
