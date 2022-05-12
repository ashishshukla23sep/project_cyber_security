
import pickle
from utlis.tuning import get_best_model
from utlis.logger import app_logger


class Model_Train:
    def model_train(self,x_train,y_train,x_val,y_val,path,logger_path):
        """
        Name:Model_train
        Description:Training model
        Input_parameter:Traing and validation data,Model path,logging path
        Output:saves the best performed model
        
        """
        try:
            self.logger=app_logger()
            file=open(logger_path/'training.txt','a+')
            obj=get_best_model(x_train,y_train,x_val,y_val,logger_path=logger_path)
            print("TRAINING STARTED for ")
            self.logger.log(file,"model_train:Model_train:  !!!!Training Started")
            model,model_name=obj.call_hyper_tunner_objective()
            self.logger.log(file,"model_train:Model_train:  !!!!Training Completed")
            model.fit(x_train,y_train)
            with open(path/'{}.pkl'.format(model_name),'ab') as f:
                    pickle.dump(model,f)
            self.logger.log(file,"model_train:Model_train:  {}.pkl model saved in data\models folder".format(model_name))
            file.close()   
        except Exception as e:
            file=open(logger_path/'training.txt','a+')
            self.logger.log(file,"model_train:Model_train:  {}".format(e))
            file.close()

        