import json
import pandas as pd
from utlis.logger import app_logger


class validate:
    def __init__(self,data,logger_path,train_path) -> None:
        self.data=data
        self.logger_path=logger_path
        self.train_path=train_path
        self.logger=app_logger()
    def validate_column_name_length(self):

        """
        Name:validate_column_name_length
        Description:To validate the number of columns and name of columns

        Output:Training data and [True False] ,True:Validated,False:Not validated
        Owner:Ashish shukla
        version:1.0
        Revision:0
        
        """
        try:
            print("validation started")
            with open('train.json','r') as f:
                column=json.load(f)
                
                f.close()
            column=dict(column)
            # print(column['columns'])
            sample_df=pd.DataFrame(column['columns'],index=[i for i in range(len(column['columns']))])
            raw_df=self.data
            diff=sample_df.columns.difference(raw_df.columns)
            print(len(diff))
            file=open(self.logger_path/'Pred_data_validation.txt','a+')
            if len(diff)==0:
                self.logger.log(file,"validat_Pred_data:Number of columns and name matches")
                raw_df.to_csv(self.train_path/'xtest.csv')
                file.close()
                print("Pred_data_verified and saved in /data/validated_train_data folder")
                return raw_df,True
            else:
                print("Pred_data_verified and saved in /data/Raw_data")
                self.logger.log(file,"validate_raw_data:Number of columns and name doesn't matches please check the csv file in data/Raw_data")
                file.close()
                return raw_df,False
        except Exception as e:

            file=open(self.logger_path/'Pred_data_validation.txt','a+')
            self.logger.log(file,"validate_raw_data:{}".format(e))
            file.close()


                
            





