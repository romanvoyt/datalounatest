import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tabpfn import TabPFNClassifier
from torch.utils.data import DataLoader

# local packages
from neuralnet.binary_feedforward_net import BinaryNN, TrainData, TestData, binary_acc
from preprocessing.preprocessing import preprocess_data
from utils.visualization import show_roc_auc_curve


def build_model(best_alg, x_train, y_train, x_test, kf, ntrain, ntest, nclass, nfolds):
    Xr_train = np.zeros((ntrain, nclass))
    Xr_test = np.zeros((ntest, nclass))
    tr_ind = np.arange(ntrain)
    for i, (ttrain, ttest) in enumerate(kf.split(tr_ind)):
        clf = best_alg
        clf.fit(x_train.iloc[ttrain], y_train.iloc[ttrain])
        sc = clf.score(x_train.iloc[ttest], y_train.iloc[ttest])
        print(f'{i} accuracy {sc:.4f}')
        Xr_train[ttest] = clf.predict_proba(x_train.iloc[ttest])
        Xr_test += clf.predict_proba(x_test) / nfolds

    return Xr_train, Xr_test


def train_and_test_model(model, x_train, y_train, x_test, y_test, kf, ntrain, ntest, nclass, nfolds, labels):
    pred_train, pred_test = build_model(model, x_train, y_train, x_test, kf, ntrain, ntest, nclass,
                                        nfolds)
    print(pred_test.shape)
    print(pred_train.shape)
    print(pred_train[:, 1].shape)
    thresholds = np.linspace(0.01, 0.9, 100)
    f1_sc = np.array([f1_score(y_train, pred_train[:, 1] > thr) for thr in thresholds])
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_sc, linewidth=4)
    plt.ylabel("F1 score", fontsize=18)
    plt.xlabel("Threshold", fontsize=18)
    best_lr = thresholds[f1_sc.argmax()]
    show_accuracy(pred_train[:, 1], y_train, labels, best_lr, nclass)
    show_accuracy(pred_test[:, 1], y_test, labels, best_lr, nclass)


def show_accuracy(Xr, y, labels, best, nclass):
    pred = []
    for x in Xr:
        if x > best:
            pred.append(1)
        else:
            pred.append(0)
    print(f'pred = {pred}')
    print(classification_report(y, pred, target_names=labels, digits=4, zero_division=True))
    print(confusion_matrix(y, pred, labels=range(nclass)))

    show_roc_auc_curve(y, pred)


def train_test_xgboost(train, players, test):
    train_df, test_df = preprocess_data(train, players, test)
    y = train_df['who_win']
    train_df.drop(['who_win'], inplace=True, axis=1)
    X = train_df

    print(f'X data = {X}')
    print(f'y data = {y}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337)

    def objective(trial):
        x = X_train
        y = y_train.values.ravel()

        max_depth = trial.suggest_int('xgb_max_depth', 2, 64, log=True)
        max_leaves = trial.suggest_int('xgb_max_leaves', 5, 20)
        n_estimators = trial.suggest_int('xgb_n_estimators', 100, 200)
        learning_rate = trial.suggest_float('xgb_learning_rate', 0.001, 0.1)
        gamma = trial.suggest_float('xgb_gamma', 1, 9)

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_leaves=max_leaves,
            gamma=gamma
        )
        score = cross_val_score(xgb_model, x, y, cv=5).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print(f'best score = {trial.value}')
    print(f'best params: ')
    for key, value in trial.params.items():
        print(f'{key} {value}')

    optuna.visualization.plot_param_importances(study)

    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    nclass = 2
    nfolds = 10
    kf = KFold(n_splits=nfolds, random_state=1337, shuffle=True)
    labels = ['Team1_win', 'Team2_win']

    xgb_clf = xgb.XGBClassifier(max_depth=2, max_leaves=10, n_estimators=150, learning_rate=0.02, gamma=2.8)
    train_and_test_model(xgb_clf, X_train, y_train, X_test, y_test, kf, ntrain, ntest, nclass, nfolds, labels)


def train_test_catboost(train, players, test):
    train_df, test_df = preprocess_data(train, players, test)
    y = train_df['who_win']
    train_df.drop(['who_win'], inplace=True, axis=1)
    X = train_df

    print(f'X data = {X}')
    print(f'y data = {y}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337)

    def objective(trial):
        x = X_train
        y = y_train.values.ravel()

        iterations = trial.suggest_int('catboost_iterations', 10, 100)
        learning_rate = trial.suggest_float('catboost_learning_rate', 0.001, 0.1)
        early_stopping_rounds = trial.suggest_int('catboost_early_stopping_rounds', 1, 10)
        depth = trial.suggest_int('catboost_depth', 1, 10)
        l2_leaf_reg = trial.suggest_int('catboost_l2_leaf_reg', 1, 10)
        random_strength = trial.suggest_float('catboost_random_strength', 0.01, 10)

        cat_model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,

        )
        score = cross_val_score(cat_model, x, y, cv=5).mean()
        return score

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=30)
    # trial = study.best_trial
    # print(f'best score = {trial.value}')
    # print(f'best params: ')
    # for key, value in trial.params.items():
    #     print(f'{key} {value}')
    #
    # optuna.visualization.plot_param_importances(study)

    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    nclass = 2
    nfolds = 10
    kf = KFold(n_splits=nfolds, random_state=1337, shuffle=True)
    labels = ['Team1_win', 'Team2_win']

    cat_clf = CatBoostClassifier(iterations=70, learning_rate=0.05,
                                 early_stopping_rounds=10, depth=10, l2_leaf_reg=5, random_strength=6)
    train_and_test_model(cat_clf, X_train, y_train, X_test, y_test, kf, ntrain, ntest, nclass, nfolds, labels)


def train_test_nn(train, players, test, epochs=100, batch_size=64, lr=0.001):
    train_df, test_df = preprocess_data(train, players, test)
    y = train_df['who_win']
    train_df.drop(['who_win'], inplace=True, axis=1)
    X = train_df

    print(f'X data = {X}')
    print(f'y data = {y}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    train_data = TrainData(torch.FloatTensor(X_train_scaled),
                           torch.FloatTensor(y_train.values.ravel()))

    test_data = TestData(torch.FloatTensor(X_test_scaled))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = BinaryNN()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for e in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e + 0:03} | Loss: {epoch_loss / len(train_loader):.5f} '
              f'| Acc: {epoch_acc / len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    show_roc_auc_curve(y_test, y_pred_list)

    print(classification_report(y_test, y_pred_list))
    print(confusion_matrix(y_test, y_pred_list))


def train_test_tabpfn(train, players, test):
    train_df, test_df = preprocess_data(train, players, test)
    y = train_df['who_win']
    train_df.drop(['who_win'], inplace=True, axis=1)
    X = train_df

    lsvc = LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=25000).fit(X, y.values.ravel())

    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y.values.ravel(), test_size=0.33, random_state=1337)

    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
    classifier.fit(X_train, y_train)

    y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

    print('Accuracy', accuracy_score(y_test, y_eval))

    show_roc_auc_curve(y_test, y_eval)

    print(f'test split data: \n')
    print(classification_report(y_test, y_eval))
    print(confusion_matrix(y_test, y_eval))

    test_df.drop(['index'], inplace=True, axis=1)
    val_df = model.fit_transform(test_df)

    y_eval, p_eval = classifier.predict(val_df, return_winning_probability=True)

    predictions = pd.DataFrame(y_eval, columns=['y_hat'])
    predictions.to_csv('predictions.csv')


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    players = pd.read_csv('data/players_feats.csv')

    train_test_xgboost(train, players, test)
    train_test_nn(train, players, test)
    train_test_tabpfn(train, players, test)
    train_test_catboost(train, players, test)