from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



#Solo sirve para calcular el tiempo que lleva el random search
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#Recibe el algoritmo a tunear y una lista de sus hiperparametros
def random_search(x, y, algorithm, params):

    folds = 5
    param_comb = 40   # folds * param_comb es la cantidad total de combinaciones que va a hacer, es dependiente del tiempo.
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)


    random_search = RandomizedSearchCV(algorithm, param_distributions=params, n_iter=param_comb, scoring='f1', n_jobs=4, cv=skf.split(x,y), verbose=3, random_state=1001 )

    
    start_time = timer(None) 
    random_search.fit(x, y)
    timer(start_time)   

    print('\n Best f1 score with %d-folds and %d combinations of hyperparameters:' % (folds, param_comb))
    print(random_search.best_score_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
