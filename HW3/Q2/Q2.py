import numpy as np
from ISLP import load_data
import pandas as pd
from ISLP.models import summarize
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import cast

training_set = pd.read_csv('training_set.csv')
testing_set = pd.read_csv('testing_set.csv')

def num_summary(x: pd.Series) -> str:
    mean = x.mean()
    std = x.std()
    median = x.median()
    minimum = x.min()
    maximum = x.max()
    return f"Mean: {mean}, Std: {std}, Median: {median}, Min: {minimum}, Max: {maximum}"
    

def PartA():
    loyalty = training_set['LoyalCH']
    special = training_set['SpecialCH']
    price_diff = training_set['PriceDiff']
    
    print("LoyalCH Summary:")
    print(num_summary(loyalty))

    print("SpecialCH Summary:")
    print(num_summary(special))

    print("PriceDiff Summary:")
    print(num_summary(price_diff))

def PartB():
    correlation_matrix = training_set[['LoyalCH', 'SpecialCH', 'PriceDiff']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

def PartC():
    X_train = training_set[['LoyalCH', 'SpecialCH', 'PriceDiff']]
    X_train = sm.add_constant(X_train)
    y_train = (training_set['Purchase'] == 'CH').astype(int)

    model = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    summary = model.summary()
    print("GLM Summary:")
    print(summary)
    
def PartD():
    X_train = training_set[['LoyalCH', 'SpecialCH', 'PriceDiff']]
    X_train = sm.add_constant(X_train)
    y_train = (training_set['Purchase'] == 'CH').astype(int)

    model = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    
    # predict probabilities on Train and classify with threshold 0.5
    probs = model.predict(X_train)
    preds = (probs >= 0.5).astype(int)

    # confusion matrix and metrics
    cm = confusion_matrix(y_train, preds)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_train)
    accuracy = (tp + tn) / total
    percent_fp = (fp / total) * 100
    percent_fn = (fn / total) * 100

    print("Confusion matrix (Train):")
    print(cm)
    print(f"Overall fraction correct (accuracy): {accuracy:.4f}")
    print(f"Percent false positives (of all observations): {percent_fp:.2f}%")
    print(f"Percent false negatives (of all observations): {percent_fn:.2f}%")

def PartE():
    X_train = training_set[['LoyalCH', 'PriceDiff']]
    X_train = sm.add_constant(X_train)
    y_train = (training_set['Purchase'] == 'CH').astype(int)

    x_testing = testing_set[['LoyalCH', 'PriceDiff']]
    x_testing = sm.add_constant(x_testing)
    y_test = (testing_set['Purchase'] == 'CH').astype(int)

    model = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    
    # predict probabilities on Train and classify with threshold 0.5
    probs = model.predict(x_testing)
    preds = (probs >= 0.5).astype(int)

    # confusion matrix and metrics
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    accuracy = (tp + tn) / total
    percent_fp = (fp / total) * 100
    percent_fn = (fn / total) * 100

    print("Confusion matrix (Test):")
    print(cm)
    print(f"Overall fraction correct (accuracy): {accuracy:.4f}")
    print(f"Percent false positives (of all observations): {percent_fp:.2f}%")
    print(f"Percent false negatives (of all observations): {percent_fn:.2f}%")
    
def PartF():
    X_train = training_set[['LoyalCH', 'PriceDiff']]
    y_train = (training_set['Purchase'] == 'CH').astype(int)

    X_test = testing_set[['LoyalCH', 'PriceDiff']]
    y_test = (testing_set['Purchase'] == 'CH').astype(int)

    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    accuracy = (tp + tn) / total
    percent_fp = (fp / total) * 100
    percent_fn = (fn / total) * 100

    print("Confusion matrix (Test):")
    print(cm)
    print(f"Overall correct: {accuracy:.4f}")
    print(f"False Positive: {percent_fp:.2f}%")
    print(f"False Negative: {percent_fn:.2f}%")

def PartG():
    X_train = training_set[['LoyalCH', 'PriceDiff']].copy()
    y_train = (training_set['Purchase'] == 'CH').astype(int)

    X_test = testing_set[['LoyalCH', 'PriceDiff']].copy()
    y_test = (testing_set['Purchase'] == 'CH').astype(int)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    ks = [5, 50, 150]
    models = {}
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_s, y_train)
        models[k] = knn

    k_eval = 5
    preds = models[k_eval].predict(X_train_s)

    cm = confusion_matrix(y_train, preds)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_train)
    accuracy = (tp + tn) / total
    percent_fp = (fp / total) * 100
    percent_fn = (fn / total) * 100

    print(f"k = 5 Results:")
    print("Confusion matrix (Train):")
    print(cm)
    print(f"Overall correct: {accuracy:.4f}")
    print(f"False Positive: {percent_fp:.2f}%")
    print(f"False Negative: {percent_fn:.2f}%")
    
    
    k_eval = 50
    preds = models[k_eval].predict(X_test_s)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    accuracy = (tp + tn) / total
    percent_fp = (fp / total) * 100
    percent_fn = (fn / total) * 100

    print(f"k = 50 Results:")
    print("Confusion matrix (Test):")
    print(cm)
    print(f"Overall correct: {accuracy:.4f}")
    print(f"False Positive: {percent_fp:.2f}%")
    print(f"False Negative: {percent_fn:.2f}%")
    
    
    k_eval = 150
    preds = models[k_eval].predict(X_train_s)

    cm = confusion_matrix(y_train, preds)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_train)
    accuracy = (tp + tn) / total
    percent_fp = (fp / total) * 100
    percent_fn = (fn / total) * 100

    print(f"k = 150 Results:")
    print("Confusion matrix (Train):")
    print(cm)
    print(f"Overall correct: {accuracy:.4f}")
    print(f"False Positive: {percent_fp:.2f}%")
    print(f"False Negative: {percent_fn:.2f}%")
    
def PartH():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Features for all models
    X_train = training_set[['LoyalCH', 'PriceDiff']]
    y_train = (training_set['Purchase'] == 'CH').astype(int)
    X_test = testing_set[['LoyalCH', 'PriceDiff']]
    y_test = (testing_set['Purchase'] == 'CH').astype(int)
    
    # Scaled features for KNN
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    plt.figure(figsize=(10, 8))
    
    # Model 1: Logistic Regression (GLM) from Part E
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    glm = sm.GLM(y_train, X_train_const, family=sm.families.Binomial()).fit()
    probs_glm = glm.predict(X_test_const)
    fpr_glm, tpr_glm, _ = roc_curve(y_test, probs_glm)
    auc_glm = roc_auc_score(y_test, probs_glm)
    plt.plot(fpr_glm, tpr_glm, label=f'Logistic Regression (AUC={auc_glm:.3f})')
    
    # Model 2: Naive Bayes from Part F
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    probs_nb = nb.predict_proba(X_test)[:, 1]
    fpr_nb, tpr_nb, _ = roc_curve(y_test, probs_nb)
    auc_nb = roc_auc_score(y_test, probs_nb)
    plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc_nb:.3f})')
    
    # Models 3, 4, 5: KNN with k=5, k=50, k=150
    auc_knn = {}
    for k in [5, 50, 150]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_s, y_train)
        probs_knn = knn.predict_proba(X_test_s)[:, 1]
        fpr_knn, tpr_knn, _ = roc_curve(y_test, probs_knn)
        auc_knn[k] = roc_auc_score(y_test, probs_knn)
        plt.plot(fpr_knn, tpr_knn, label=f'KNN k={k} (AUC={auc_knn[k]:.3f})')
    
    # Plot details
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Classification Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    
    # Find best model
    all_aucs = {
        'Logistic Regression': auc_glm,
        'Naive Bayes': auc_nb,
        'KNN (k=5)': auc_knn[5],
        'KNN (k=50)': auc_knn[50],
        'KNN (k=150)': auc_knn[150]
    }
    
    best_model = max(all_aucs.items(), key=lambda x: x[1])
    
    print("AUC values:")
    for model, auc in all_aucs.items():
        print(f"{model}: {auc:.4f}")
    print(f"\nBest model: {best_model[0]} with AUC = {best_model[1]:.4f}")