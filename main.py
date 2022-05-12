
import pandas as pd
from pathlib import Path
from preprocessing_data import Preprocessing_raw
from model_train import Model_Train
from validate_raw_data import validate
import pickle
import os
import shutil
from utlis.logger import app_logger
from Validate_predict_data import validate_pred

class main(object):

    def __init__(self) -> None:
        self.PATH = Path('data/')
        self.TRAIN_PATH = self.PATH/'Validated_train_data'
        self.DATAPROCESSORS_PATH = self.PATH/'dataprocessors'
        self.MODELS_PATH = self.PATH/'models'
        self.logger_path = self.PATH/'messages'
        self.intial_data=self.PATH/'Raw_data'
        self.predict_data=self.PATH/'predict_data'
        self.Valid_predict_data=self.PATH/'Valid_predict_data'
        self.logger=app_logger()

    def create_folder(self):
        """
        This will create the all the folder mention in constructor

        """
        print("creating directory structure...")
        (self.PATH).mkdir(exist_ok=True)
        (self.TRAIN_PATH).mkdir(exist_ok=True)
        (self.MODELS_PATH).mkdir(exist_ok=True)
        (self.DATAPROCESSORS_PATH).mkdir(exist_ok=True)
        (self.logger_path).mkdir(exist_ok=True)
        (self.intial_data).mkdir(exist_ok=True)
        (self.predict_data).mkdir(exist_ok=True)
        (self.Valid_predict_data).mkdir(exist_ok=True)
        


    def load_data(self,PATH2=None):
        df=pd.read_csv(self.intial_data/'train.csv')
        df,status=validate(df,self.logger_path,self.TRAIN_PATH).validate_column_name_length()
        if status:
            pre_object=Preprocessing_raw(df,self.logger_path)
            x_train,y_train,x_val,y_val=pre_object.split_data()
            return x_train,y_train,x_val,y_val,True
        else:
            return None,None,None,None,False


        

    def train_model(self):

        try:
            self.create_folder()
            shutil.rmtree(self.logger_path)
            self.create_folder()
            x_train,y_train,x_val,y_val,status=self.load_data()
            file=open(self.logger_path/'training.txt','a+')
            if status:
                # x_test.to_csv(self.predict_data/'xtest.csv',index=False)
                # y_test.to_csv(self.predict_data/'ytest.csv',index=False)
                
                md=Model_Train()
                md.model_train(x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,path=self.MODELS_PATH,logger_path=self.logger_path)
                self.logger.log(file,"main:train_mode:  !!!!Training Started")
                file.close()
            else:
                self.logger.log(file,"main:train_model---  No Valid file found ,please check the raw data and Raw_validation.txt log")
                file.close()
        except Exception as e:
            file=open(self.logger_path/'training.txt','a+')
            self.logger.log(file,"main:train_model  {}".format(e))
            file.close()
            
    def predict(self):
        
       
        try:
            self.create_folder()
            shutil.rmtree(self.logger_path)
            self.create_folder()
            file=open(self.logger_path/'predict.txt','a+')
            data=pd.read_csv(self.predict_data/'xtest.csv')

            df,status=validate_pred(data,self.logger_path,self.Valid_predict_data).validate_column_name_length()
            if status:
                if len(os.listdir(self.MODELS_PATH))==0:#if no model exist then it will start training with raw data
                    self.logger.log(file,"MAIN:PREDICT:  Opps didn't find any pre trained model....starting new training")
                    self.train_model()
                    self.predict(data)
                    file.close()
                else:
                    files = [x for x in os.listdir(self.MODELS_PATH) if x.endswith("pkl") ]
                    paths = [os.path.join('data\models', basename) for basename in files]
                    newest = max(paths , key = os.path.getatime)
                    print("Recently modified Docs",newest)
                    model=pickle.load(open(newest,'rb'))
                    print(model.predict(df))
                    y_pred=model.predict(data)
                    pd.DataFrame(y_pred).to_csv(self.predict_data/'y_pred.csv')
                    self.logger.log(file,"MAIN:PREDICT: Prediction completed and saved in data/predict_data/ypred.csv")
                    file.close()
                    return y_pred
            else:
                file=open(self.logger_path/'predict.txt','a+')
                self.logger.log(file,"MAIN:PREDICT: Please check the csv file in data/predict_data folder")
                file.close()
        except Exception as e:
            file=open(self.logger_path/'predict.txt','a+')
            self.logger.log(file,"MAIN:PREDICT: --{}".format(e))
            file.close()


train=main()
y_pred=train.predict()

