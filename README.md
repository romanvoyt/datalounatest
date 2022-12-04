# datalounatest
Test forecast for datalouna

# Описание директорий

<b>data</b> - папка с базовыми файлами и предобработанным датасетом \
<b>neuralnet</b> - нейронная сеть на pytroch'е \
<b>preprocessing</b> - препроцессинг и feature engineering \
<b>utils</b> - вспомогательные функции для визуализации 

# Описание основного скрипта

train_and_test_pipeline - в этом пайплайне тестируется 4 алгоритма (xgboost, catboost, feedforward nn и [tabpfn](https://arxiv.org/abs/2207.01848])

Предикты сделал только по tabpfn, так как он лучше всех справляется на тренировочной выборке.
