from packages import *

def model_lstm(hyper, X_train, y_train2, X_val, y_val2):
    
    num_classes = 3
    Adam=optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=True)
    
    #Cleaning session
    tensorflow.keras.backend.clear_session()

    #Hyper
    hyper=hyper.iloc[np.argsort(hyper.loc[:,'score']),:].tail(1).reset_index()
    neuron=hyper.loc[0,'neurons'] 
    lamb1=hyper.loc[0,'lamb1'] 
    lamb2=hyper.loc[0,'lamb2']  

    #Model for classification
    inputs = Input(shape=np.shape(X_train)[1:]) 
    time_layer = TimeDistributed(Lambda(lambda x: x))(inputs) 
    lstm = LSTM(neuron, kernel_regularizer=regularizers.l1_l2(lamb1, lamb2))(time_layer)
    soft = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(lamb1, lamb2))(lstm)

    #Final model
    model = Model(inputs, soft)

    #Compiling
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])

    #Running
    model.fit(X_train, y_train2, epochs=50,
                                 batch_size=500,
                                 shuffle=True,
                                 verbose=False,
                                 validation_data=(X_val, y_val2))
    return model

def model_mlp(hyper, X_train, y_train2, X_val, y_val2):
    
    num_classes = 3
    Adam=optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=True)

    
    #Cleaning session
    tensorflow.keras.backend.clear_session()

    #Hyper 
    hyper=hyper.iloc[np.argsort(hyper.loc[:,'score']),:].tail(1).reset_index()
    neuron=hyper.loc[0,'neurons'] 
    lamb1=hyper.loc[0,'lamb1'] 
    lamb2=hyper.loc[0,'lamb2']

    #LogReg
    inputs = Input(shape=np.shape(X_train)[1:])
    h = Dense(neuron, activation='relu', kernel_regularizer=regularizers.l1_l2(lamb1, lamb2))(inputs)
    soft = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(lamb1, lamb2))(h)
    model = Model(inputs, soft)

    #Compiling
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])

    #Running
    model.fit(X_train, y_train2, epochs=50,
                                                  batch_size=500,
                                                  shuffle=True,
                                                  verbose=False,
                                                  validation_data=(X_val, y_val2))
    return model


def model_xgboost(hyper, X_train, y_train, X_val, y_val, random_seed):
    
    #Hyper 
    n_estimators=500
    hyper=hyper.iloc[np.argsort(hyper.loc[:,'score']),:].tail(1).reset_index()
    gamma=hyper.loc[0,'gamma'] 
    learning_rate=hyper.loc[0,'learning_rate'] 
    max_depth=hyper.loc[0,'max_depth'] 
    reg_lambda=hyper.loc[0,'reg_lambda'] 
    gamma=hyper.loc[0,'gamma'] 


    model = xgb.XGBClassifier(objective = 'multi:softmax',
                                  gamma = gamma,
                                  learning_rate = learning_rate,
                                  max_depth = max_depth,
                                  reg_lambda = reg_lambda,
                                  n_estimators=  n_estimators,
                                  use_label_encoder=False,
                                  seed=random_seed,
                                  n_jobs=-1)

    model.fit(X_train, y_train,
                  verbose=False,
                  early_stopping_rounds=15,
                  eval_metric='merror',
                  eval_set=[(X_val, y_val)])
    return model



def bootstrap(X_test,y_test, model, nn=True, B=100):
    #Creating dictionary to store results
    out={}
    out['accuracy']=[]
    out['macro avg']={}
    out['macro avg']['f1-score']=[]
    out['macro avg']['recall']=[]
    out['macro avg']['precision']=[]
    out['weighted avg']={}
    out['weighted avg']['f1-score']=[]
    out['weighted avg']['recall']=[]
    out['weighted avg']['precision']=[]

    #Running Bootstrap on the test set
    for b in tqdm(range(B)):
        ind = np.random.choice(range(y_test.shape[0]),y_test.shape[0])
        X_test_boot, y_test_boot = X_test[ind,:], y_test[ind]

        y_pred=model.predict(X_test_boot)
        
        if nn:
            y_pred=np.argmax(y_pred,axis=1)
            report=classification_report(y_test_boot, y_pred, labels=[0, 1, 2], output_dict=True)
        else:
            report=classification_report(y_test_boot, y_pred, labels=[0, 1, 2], output_dict=True)

        out['accuracy'].append(report['accuracy'])
        out['macro avg']['f1-score'].append(report['macro avg']['f1-score'])
        out['macro avg']['recall'].append(report['macro avg']['recall'])
        out['macro avg']['precision'].append(report['macro avg']['precision'])
        out['weighted avg']['f1-score'].append(report['weighted avg']['f1-score'])
        out['weighted avg']['recall'].append(report['weighted avg']['recall'])
        out['weighted avg']['precision'].append(report['weighted avg']['precision'])

    #Preparing output
    y_pred=model.predict(X_test)
    
    if nn:
        y_pred=np.argmax(y_pred,axis=1)
        report=classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)
    else:
        report=classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)

    out['accuracy'] = [report['accuracy'], np.std(out['accuracy'])]
    out['macro avg']['f1-score'] = [report['macro avg']['f1-score'], np.std(out['macro avg']['f1-score'])] 
    out['macro avg']['recall'] = [report['macro avg']['recall'], np.std(out['macro avg']['recall'])] 
    out['macro avg']['precision'] = [report['macro avg']['precision'], np.std(out['macro avg']['precision'])] 
    out['weighted avg']['f1-score'] = [report['weighted avg']['f1-score'], np.std(out['weighted avg']['f1-score'])] 
    out['weighted avg']['recall'] = [report['weighted avg']['recall'], np.std(out['weighted avg']['recall'])] 
    out['weighted avg']['precision'] = [report['weighted avg']['precision'], np.std(out['weighted avg']['precision'])]
    
    return out