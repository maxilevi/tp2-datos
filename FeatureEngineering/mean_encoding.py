def mean_encoding_smooth(xtrain,ytrain,alpha):
    #alpha es el smooth a usar, si es 0 genera que sea un mean_encoding(causa overfitting porque se filtra el label)
    #y si es muy elevado se va a tener el mismo valor para todas las keywords.

    merged=xtrain.merge(ytrain,on='id')
    global_mean=merged['target'].mean()
    rows_range=len(merged)
    merged['mean_keyword']=merged.groupby('keyword')['target'].transform('mean')
    xtrain['mean_encode']=((rows_range*merged['mean_keyword'])+global_mean*alpha)/(rows_range+alpha)

    return xtrain
