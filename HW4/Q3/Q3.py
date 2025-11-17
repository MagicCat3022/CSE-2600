import numpy as np
from ISLP import load_data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier as GBM
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data.csv')

def PartA():
    random_state = 67
    
    s7 = data['Store7']
    s7.replace("Yes", 1 , inplace=True)
    s7.replace("No", 0 , inplace=True)
    
    data['Store7'] = s7

    train = data.sample(n=800, random_state=random_state)
    test = data.drop(train.index)
    
    return train, test

def PartB():
    train, test = PartA()

    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)

    y_train, labels = pd.factorize(train['Purchase'])
    print("Class labels:", labels)

    DTC = DecisionTreeClassifier(max_leaf_nodes=6, random_state=67)
    DTC.fit(X_train, y_train)

    y_pred = DTC.predict(X_train)

    cm = confusion_matrix(y_train, y_pred)
    error = 1.0 - accuracy_score(y_train, y_pred)

    print("Confusion matrix:")
    print(cm)
    print(f"Training misclassification error: {error:.4f}")

    return DTC, cm, error

def PartC():
    DTC, cm, error = PartB()

    train, test = PartA()
    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    tree.plot_tree(
        DTC,
        feature_names=list(X_train.columns),
        class_names=DTC.classes_.astype(str),
        filled=True,
        ax=ax
    )
    ax.set_title('Decision Tree (max 6 leaf nodes)')
    plt.tight_layout()

    fig.savefig('PartC.png')

    plt.show()
    return fig

def PartD():
    train, test = PartA()
    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)

    y_train, labels = pd.factorize(train['Purchase'])

    clf = DecisionTreeClassifier(max_leaf_nodes=6, random_state=67)
    clf.fit(X_train, y_train)

    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    error = 1.0 - accuracy_score(y_test, y_pred)

    print("Test confusion matrix:")
    print(cm)
    print(f"Test misclassification error: {error:.4f}")

    return cm, error

def PartE():
    train, test = PartA()

    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)

    y_train, labels = pd.factorize(train['Purchase'])
    print("Class labels:", labels)

    DTC = DecisionTreeClassifier(max_leaf_nodes=8, random_state=67)
    DTC.fit(X_train, y_train)

    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values

    y_pred = DTC.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    error = 1.0 - accuracy_score(y_test, y_pred)

    print("Test confusion matrix:")
    print(cm)
    print(f"Test misclassification error: {error:.4f}")


    return DTC, cm, error

def PartG():
    train, test = PartA()

    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)

    y_train, labels = pd.factorize(train['Purchase'])

    param_grid = {'max_leaf_nodes': list(range(2, 25))}
    DTC = DecisionTreeClassifier(random_state=67)
    grid = GridSearchCV(DTC, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_params_['max_leaf_nodes']
    best_score = grid.best_score_
    best_err = 1.0 - best_score
    print(f"Best max_leaf_nodes: {best} (CV accuracy={best_score:.4f}, CV misclassification={best_err:.4f})")

    best_clf = DecisionTreeClassifier(max_leaf_nodes=best, random_state=67)
    best_clf.fit(X_train, y_train)

    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values

    y_pred = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    test_err = 1.0 - accuracy_score(y_test, y_pred)

    print("Test confusion matrix (best model):")
    print(cm)
    print(f"Test misclassification error (best model): {test_err:.4f}")

    PartD()

    return grid, best_clf, cm, test_err

def PartH():
    train, test = PartA()
    
    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)
    y_train, labels = pd.factorize(train['Purchase'])
    
    RFC = RandomForestClassifier(n_estimators=3500, random_state=67, max_features=5, min_samples_split=2, min_samples_leaf=4, n_jobs=-1)
    RFC.fit(X_train, y_train)
    
    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values
    y_pred = RFC.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    error = 1.0 - accuracy_score(y_test, y_pred)
    print("Random Forest Test confusion matrix:")
    print(cm)
    print(f"Random Forest Test misclassification error: {error:.4f}")
    
    return RFC, cm, error
    
def PartJ():
    train, test = PartA()
    
    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)
    y_train, labels = pd.factorize(train['Purchase'])
    
    RFC = RandomForestClassifier(n_estimators=3500, random_state=67, max_features=5, min_samples_split=2, min_samples_leaf=4, n_jobs=-1)
    RFC.fit(X_train, y_train)
    
    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values
    y_preds = RFC.predict_proba(X_test)[:, 0]
    y_true = y_test
    n_bins = 10
    
    print(label_map)
    
    data = pd.DataFrame({'y_true': y_true, 'y_preds': y_preds})
    
    # Sort the DataFrame by predicted values
    data = data.sort_values(by='y_preds', ascending=False)

    # Create bins based on predicted values
    data['bin'] = pd.qcut(data['y_preds'], n_bins, labels=False)

    # Calculate lift
    lift_data = data.groupby('bin').agg(
        total_count=('y_true', 'count'),
        actual_mean=('y_true', 'mean'),
        predicted_mean=('y_preds', 'mean')
    ).reset_index()

    # Calculate expected mean (mean of actual values)
    expected_mean = data['y_true'].mean()
    
    # Calculate lift
    lift_data['lift'] = lift_data['actual_mean'] / expected_mean

    # Plot the lift chart
    plt.figure(figsize=(10, 6))
    plt.plot(lift_data['bin'], lift_data['lift'], marker='o', label='Lift')
    plt.axhline(y=1, color='r', linestyle='--', label='Random Guessing Lift (1)')
    plt.title('Lift Chart for Regression Model')
    plt.xlabel('Bins')
    plt.ylabel('Lift')
    plt.xticks(lift_data['bin'])
    plt.legend()
    plt.grid()
    plt.savefig('PartJ.png')
    plt.show()

def PartK():
    train, test = PartA()
    
    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)
    y_train, labels = pd.factorize(train['Purchase'])

    GBM_model = GBM(n_estimators=350, random_state=67, max_features=10, learning_rate=0.05, max_depth=3, subsample=1, min_samples_split=2)
    GBM_model.fit(X_train, y_train)
    
    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values
    
    y_predict = GBM_model.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    error = 1.0 - accuracy_score(y_test, y_predict)
    
    print("GBM Test confusion matrix:")
    print(cm)
    print(f"GBM Test misclassification error: {error:.4f}")
    
    return GBM_model, cm, error

def PartM():
    RFM, cm, error = PartH()
    GBM_model, cm, error = PartK()
    train, test = PartA()
    
    X_train = train.drop(columns=['Purchase'])
    X_train = pd.get_dummies(X_train, drop_first=True)
    y_train, labels = pd.factorize(train['Purchase'])
    X_test = test.drop(columns=['Purchase'])
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_test = test['Purchase'].map(label_map).values
    y_true = y_test
    
    y_pred_model1 = RFM.predict_proba(X_test)[:, 0]
    y_pred_model2 = GBM_model.predict_proba(X_test)[:, 0]
    model1_name = "Random Forest"
    model2_name = "GBM"
    n_bins = 10
    
    # Build combined dataframe
    data = pd.DataFrame({
        'y_true': y_true,
        'pred1': y_pred_model1,
        'pred2': y_pred_model2
    })

    # Avoid division by zero
    data = data[data['pred2'] != 0]

    # Compute ratio of predictions
    data['ratio'] = data['pred1'] / data['pred2']

    # Sort descending by ratio
    data = data.sort_values(by='ratio', ascending=False)

    # Create quantile bins based on the ratio
    data['bin'] = pd.qcut(data['ratio'], n_bins, labels=False, duplicates='drop')

    # Overall means for normalization
    overall_mean_true = data['y_true'].mean()
    overall_mean_pred1 = data['pred1'].mean()
    overall_mean_pred2 = data['pred2'].mean()

    # Aggregate lift stats for each bin
    lift_data = data.groupby('bin').agg(
        actual_mean=('y_true', 'mean'),
        model1_mean=('pred1', 'mean'),
        model2_mean=('pred2', 'mean')
    ).reset_index()

    # Compute model error in each bin
    lift_data['model1_lift'] = lift_data['model1_mean'] / lift_data['actual_mean']
    lift_data['model2_lift'] = lift_data['model2_mean'] / lift_data['actual_mean']

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lift_data['bin'], lift_data['model1_lift'], marker='s', linestyle='-', linewidth=2, label=f'{model1_name} Predicted Lift')
    plt.plot(lift_data['bin'], lift_data['model2_lift'], marker='^', linestyle='--', linewidth=2, label=f'{model2_name} Predicted Lift')

    plt.axhline(y=1, color='red', linestyle=':', label='Baseline (Lift = 1)')
    plt.title(f'Double Lift Chart by Ratio of {model1_name} / {model2_name}', fontsize=14)
    plt.xlabel(f'Bins (sorted by {model1_name} / {model2_name} ratio)')
    plt.ylabel('Lift (Mean / Overall Mean)')
    plt.xticks(np.arange(0, lift_data.shape[0]))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('PartM.png')
    plt.show()
    
PartM()