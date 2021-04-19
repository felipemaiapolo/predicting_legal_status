from packages import *

def expand_grid(dictionary, max_rows=50, random_seed=22):
       return pd.DataFrame([row for row in product(*dictionary.values())], 
                      columns=dictionary.keys()).sample(frac=1,random_state=random_seed).reset_index(drop=True).iloc[:max_rows,:]

def tune_xgboost(hyper, X_train, y_train, X_val, y_val, random_seed=22):
    
    for i in tqdm(range(np.shape(hyper)[0])): 
        n_estimators=500
        gamma=hyper.loc[i,'gamma'] 
        learning_rate=hyper.loc[i,'learning_rate'] 
        max_depth=hyper.loc[i,'max_depth'] 
        reg_lambda=hyper.loc[i,'reg_lambda'] 
        gamma=hyper.loc[i,'gamma'] 


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

        p=1-np.min(model.evals_result()["validation_0"]["merror"])

        hyper.loc[i,'n_est']=np.argmin(model.evals_result()["validation_0"]["merror"])+1
        hyper.loc[i,'score']=p
        hyper.loc[i,'lower_ci']=p-1.96*np.sqrt((p*(1-p)/np.shape(y_val)[0]))
        hyper.loc[i,'upper_ci']=p+1.96*np.sqrt((p*(1-p)/np.shape(y_val)[0]))
        
    return hyper

def tune_lstm(hyper, X_train, y_train2, X_val, y_val2):
    
    #start_time = time.time()
    num_classes = 3
    Adam=optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=True)
    
    for i in tqdm(range(np.shape(hyper)[0])):
        
        #Cleaning session
        tensorflow.keras.backend.clear_session()

        #Hyper
        neuron=hyper.loc[i,'neurons'] 
        lamb1=hyper.loc[i,'lamb1'] 
        lamb2=hyper.loc[i,'lamb2']  

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
        modelo=model.fit(X_train, y_train2, epochs=50,
                                                  batch_size=500,
                                                  shuffle=True,
                                                  verbose=False,
                                                  validation_data=(X_val, y_val2))

        p=modelo.history['val_accuracy'][-1]
        hyper.loc[i,'score']=p
        hyper.loc[i,'lower_ci']=p-1.96*np.sqrt((p*(1-p)/np.shape(y_val2)[0]))
        hyper.loc[i,'upper_ci']=p+1.96*np.sqrt((p*(1-p)/np.shape(y_val2)[0]))

        #progress
        #if i%int(np.shape(hyper)[0]/10)==0: print(round(100*i/np.shape(hyper)[0],0),"% concluded in", np.round((time.time() - start_time)/60,2),"minutes")
        #else: pass
        
    return hyper

def tune_mlp(hyper, X_train, y_train2, X_val, y_val2):
    
    #start_time = time.time()
    num_classes = 3
    Adam=optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=True)

    for i in tqdm(range(hyper.shape[0])):

        #Cleaning session
        tensorflow.keras.backend.clear_session()

        #Hyper 
        neuron=hyper.loc[i,'neurons'] 
        lamb1=hyper.loc[i,'lamb1'] 
        lamb2=hyper.loc[i,'lamb2'] 

        #LogReg
        inputs = Input(shape=np.shape(X_train)[1:])
        h = Dense(neuron, activation='relu', kernel_regularizer=regularizers.l1_l2(lamb1, lamb2))(inputs)
        soft = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(lamb1, lamb2))(h)
        model = Model(inputs, soft)

        #Compiling
        model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])

        #Running
        modelo=model.fit(X_train, y_train2, epochs=50,
                                                  batch_size=500,
                                                  shuffle=True,
                                                  verbose=False,
                                                  validation_data=(X_val, y_val2))

        p=modelo.history['val_accuracy'][-1]
        hyper.loc[i,'score']=p
        hyper.loc[i,'lower_ci']=p-1.96*np.sqrt((p*(1-p)/np.shape(y_val2)[0]))
        hyper.loc[i,'upper_ci']=p+1.96*np.sqrt((p*(1-p)/np.shape(y_val2)[0]))
        
    return hyper