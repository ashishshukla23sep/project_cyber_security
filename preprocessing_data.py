from dataclasses import dataclass
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import KNNImputer
import numpy as np
import json

from sqlalchemy import column
from utlis.logger import app_logger

class Preprocessing_raw:
    def __init__(self,data,logger_path) -> None:
        self.data=data
        self.logger_path=logger_path
        self.logger=app_logger()

    def preprocessing(self):
        """
        Name:Preprocessing
        Description:Fill the missing values if present and removet the features which has std zero

        Written by:Ashish Shukla
        version:1.0
        Revision:None
        
        """
        file=open(self.logger_path/'Preprocessing.txt','a+')
        self.logger.log(file,"Preprocessing:preprocessing --Starting Preprocessing")
        pred_sample={}
        null_columns=[i for i in self.data.columns if self.data[i].isnull().sum()>0]
        self.logger.log(file,"Preprocessing:split_data number of null columns {}".format(null_columns))
        if len(null_columns)>0:
            imput=KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            imput.fit_transform(self.data)

        std_zero_column=[i for i in self.data.columns if np.std(self.data[i])==0]
        #removing the label and std_zero columns
        pred_sample['columns_length']=len(self.data.columns)-len(std_zero_column)-1
        
        self.data.drop(columns=std_zero_column,inplace=True)
        pred_sample['columns']={i:"int" for i in self.data.drop(columns='phishing').columns}
        pred_sample['Columns_to_be_droped']=std_zero_column
        json.dump(pred_sample,open("pred.json", "w"))
        self.logger.log(file,"Preprocessing:preprocessing -- Preprocessing Completed")
        file.close()
        return self.data

    def split_data(self):
        
        processed_data=self.preprocessing()
        file=open(self.logger_path/'Preprocessing.txt','a+')
        self.logger.log(file,"Preprocessing:split_data --spliting data started")
        
        x=processed_data.drop(columns='phishing')
        y=processed_data.phishing
        sss=StratifiedShuffleSplit(n_splits=1,random_state=10,test_size=0.3)
        train_index,val_index = next(sss.split(x,y))
        x_train,x_val=x.iloc[train_index],x.iloc[val_index]
        y_train,y_val=y.iloc[train_index],y.iloc[val_index]
        
        self.logger.log(file,"Preprocessing:split_data -- Spliting completed")
        file.close()
        
        return x_train,y_train,x_val,y_val
    