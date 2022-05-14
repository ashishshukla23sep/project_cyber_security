from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,accuracy_score
from xgboost import XGBClassifier
from utlis.logger import app_logger

class get_best_model:
    def __init__(self,x_train,y_train,x_val,y_val,logger_path):
        self.x_train=x_train
        self.y_train=y_train
        self.x_val=x_val
        self.y_val=y_val
        self.logger_path=logger_path
        self.logger=app_logger()

    def tunner_XGB(self,params):

        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        classifier = XGBClassifier(**params)
        
        classifier.fit(self.x_train, self.y_train)
        
        # Applying k-Fold Cross Validation
        if len(self.y_train.unique())==1:#check accuracy score if only one label exist
            scr=cross_val_score(classifier,self.x_val,self.y_val,scoring='accuracy').mean()
            return{'loss':1-scr, 'status': STATUS_OK }
        else:
            scr=cross_val_score(classifier,self.x_val,self.y_val,scoring='roc_auc').mean()
            return{'loss':1-scr, 'status': STATUS_OK }
    

    def tunner_RFC(self,params):

        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        classifier = RandomForestClassifier(**params)
        
        classifier.fit(self.x_train, self.y_train)
        
        # Applying k-Fold Cross Validation
        if len(self.y_train.unique())==1:#check accuracy score if only one label exist
            scr=cross_val_score(classifier,self.x_val,self.y_val,scoring='accuracy').mean()
            return{'loss':1-scr, 'status': STATUS_OK }
        else:
            scr=cross_val_score(classifier,self.x_val,self.y_val,scoring='roc_auc').mean()
            return{'loss':1-scr, 'status': STATUS_OK }
    

    def call_hyper_tunner_objective(self):

        #Dict of Hyperparameter
        space_XGB= {
                    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
                    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
                    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
                    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
                    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
                    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
                }

        space_RFC= {
                    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
                    "max_depth": hp.quniform("max_depth", 1, 15,1),
                    "criterion": hp.choice("criterion", ["gini", "entropy"]),
                    }
        #Intiating the trails
        trails_XGB=Trials()
        print("XGB TUNING STARTED...")
        file=open(self.logger_path/'Hyper_parameter.txt','a+')
        self.logger.log(file,"XGB TRAINING STARTED..")
        best_XGB=fmin(  fn=self.tunner_XGB,
                        space = space_XGB, 
                        algo=tpe.suggest, 
                        max_evals=50, 
                        trials=trails_XGB
                    )
        
        print("RFC TUNING STARTED...")
        
        self.logger.log(file,"RFC TRAINING STARTED..")          
        trails_RFC=Trials()
        best_RFC=fmin(  fn=self.tunner_RFC,
                        space = space_RFC, 
                        algo=tpe.suggest, 
                        max_evals=50, 
                        trials=trails_RFC
                    )
        if min(trails_RFC.losses())<min(trails_XGB.losses()):
            self.logger.log(file,"BEST model selected is RFC with loss of {}".format(min(trails_RFC.losses())))
            file.close()
            return RandomForestClassifier(n_estimators=best_RFC["n_estimators"],
                                          max_depth=best_RFC["max_depth"],
                                          criterion=best_RFC["criterion"] ),'RFC'
        else:
            self.logger.log(file,"BEST model selected is XGB with loss of {}".format(min(trails_XGB.losses())))
            file.close()
            return XGBClassifier(n_estimators = best_XGB['n_estimators'],
                            max_depth = best_XGB['max_depth'],
                            learning_rate = best_XGB['learning_rate'],
                            gamma = best_XGB['gamma'],
                            min_child_weight = best_XGB['min_child_weight'],
                            subsample = best_XGB['subsample'],
                            colsample_bytree = best_XGB['colsample_bytree'],),'XGB'


