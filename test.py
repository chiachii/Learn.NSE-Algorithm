from LearnNSE import *


def get_next_train(df, batch, k, target_name):
    '''
    @df: data frame
    @batch: the number of training datasets you need
    @k: the kth time in the training and testing loop, where k > 0
    @target_name: the name of the target column
    '''
    # get sub-dataset of all dataset, which length equals to batch
    X_train = df[(k-1)*batch : (k)*batch]

    # X_train, y_train
    y_train = pd.DataFrame(X_train[target_name])
    X_train = X_train.drop(columns=target_name)

    return X_train, y_train

############# Training Learn++.NSE #############
# record
record = []

# number of run time
RunTime = 10 # default=10

# count the time length
time_length = len(df) # df is your dataframe
batch = int((1/RunTime)*time_length) # take 1/RunTime of all dataset

# set a model
LearnPPNSE = LearnNSE()

# framework
for k in range(1,RunTime+1):
    # at least a model inside    
    if k == 1:
        # get the dataset for training
        X_train, y_train = get_next_train(df=df, batch=batch, k=k, target_name='Label')
        # training model
        LearnPPNSE.fit(X_train, y_train)
        # re-compute voting weight
        LearnPPNSE.revoting(X_train, y_train)
        
    else:
        # get the dataset for training
        X_train, y_train = get_next_train(df=df, batch=batch, k=k, target_name='Label')
        # re-build the error distribution
        LearnPPNSE.redistribute_error_rate(X_train, y_train)
        # training model
        LearnPPNSE.fit(X_train, y_train)
        # re-compute voting weight
        LearnPPNSE.revoting(X_train, y_train)

    # testing & recording the score
    if k < RunTime:
        X_test, y_test = get_next_train(df=df, batch=batch, k=k+1, target_name='Label')
        score_B = LearnPPNSE.score(X_test, y_test)
        record.append(round(score_B, 3))
################################################