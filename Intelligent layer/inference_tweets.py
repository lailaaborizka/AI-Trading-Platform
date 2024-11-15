from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import pandas as pd 
from models.lstm import LSTMModel
from models. MLP import MLP
from models.transformer import TimeSeriesTransformer
import numpy as np
import preprocessing.preprocessing as preprocessing
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from callbacks.callbacks import PyTorchLightningPruningCallback_2
import optuna
from optuna.samplers import TPESampler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from preprocessing.visualization_module import plotTimeSeries
from lightning.pytorch.loggers import WandbLogger
import os 
import time
from pynvml import *
import random
import torch
import logging
import sys

DATA_BASE_NAME = "studies"
db = SQLAlchemy()
app = Flask(__name__)

#databseURL = f'postgresql://postgres:postgres@localhost:5432/{DATA_BASE_NAME}'

#app.config['SQLALCHEMY_DATABASE_URI'] = databseURL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.environ['WANDB_CONSOLE']="off"
# os.environ["WANDB_MODE"] = "offline"


#db.init_app(app)

SEED = 42

np.random.seed(SEED)


TEST_SIZE = 0.1
STEP_SIZE = 1

hidden_layer_size= 32
num_layers= 1
output_size=1
dropout=0.1
lr = 1e-3

test_size = 0.1
batch_size = 64
target_col_name = "close"
enc_seq_len = 7
output_sequence_length = 1 
window_size = enc_seq_len + output_sequence_length 

exogenous_vars = ['open' , 'high' , 'low', 'volume','sentiment_score'] 
# exogenous_vars = ['open' , 'low' , 'high']
# exogenous_vars = []
input_variables = [target_col_name] + exogenous_vars
target_variables = ['close']


data_name_prefix = 'finaldf_with_sentiment_TSLA.csv'
data = pd.read_csv(data_name_prefix)
# data.drop(columns=['Ticker'] , inplace=True)
# print("data loaded")
# data = data[:10000]

# preprocessing_std = preprocessing.Preprocessing(
#     prefix = data_name_prefix,
#     data=data,
#     window_size = 5,
#     target_window_size = 1,
#     step_size=STEP_SIZE,
#     input_variables=input_variables,
#     target_variables = target_variables,
#     test_size=TEST_SIZE,
#     normalize_data = 'None',
#     make_reference= True
# )

# preprocessing_std.create(scaling='std' , trading_day=True , group_by_day=True)

# training_data_std = preprocessing_std.get_train_loader(shuffle= True , batch_size = 128)
# testing_data_std= preprocessing_std.get_validation_loader(shuffle= False, batch_size = 128)

# preprocessing_minmax = preprocessing.Preprocessing(
#     prefix = data_name_prefix,
#     data=data,
#     window_size = 5,
#     target_window_size = 1,
#     step_size=STEP_SIZE,
#     input_variables=input_variables,
#     target_variables = target_variables,
#     test_size=TEST_SIZE,
#     normalize_data = 'None',
#     make_reference= True
# )


# preprocessing_minmax.create(scaling='minmax' , trading_day=True , group_by_day=True)

# training_data_minmax = preprocessing_minmax.get_train_loader(shuffle= True , batch_size = 128)
# testing_data_minmax = preprocessing_minmax.get_validation_loader(shuffle= False, batch_size = 128)


nvmlInit()
def objective(trial):


        
    # print("trial number here : " , trial.number)
    os.environ['WANDB_CONSOLE']="off"
    # os.environ["WANDB_START_METHOD"]="thread"
    # os.environ['WANDB_DISABLE_SERVICE']="flase"

    run = wandb.init(group = "DDP",mode = "offline", project='MLP-study-std-only' , name = f"MLP_{trial.number}")
    
    os.environ['WANDB_CONSOLE']="off"
    # os.environ["WANDB_START_METHOD"]="thread"
    # os.environ['WANDB_DISABLE_SERVICE']="flase"
   
#    , offline = True, project='LSTM-study-std-minmax-only' ,log_model=False , name = f"LSTM_{trial.number}"
    wandb_logger = WandbLogger(experiment = run)   
    os.environ['WANDB_CONSOLE']="off"
    # os.environ["WANDB_START_METHOD"]="thread"
    # os.environ['WANDB_DISABLE_SERVICE']="flase"
   
    normalization_options = {
        0 : 'std',
        1 : 'minmax',
    }

    CheckpointCallback_val = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name = True,
        filename = 'val_loss_epoch-{epoch}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}-{turn_around:.4f}-{positive_residuals:.4f}-{negative_residuals:.4f}-{mape:.4f}'

        )
    CheckpointCallback_turn_around = ModelCheckpoint(
            monitor="turn_around",
            mode="max",
            auto_insert_metric_name = True,
            filename = 'turn_around-{epoch}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}-{turn_around:.4f}-{positive_residuals:.4f}-{negative_residuals:.4f}-{mape:.4f}'

        )
    CheckpointCallback_positive_residuals = ModelCheckpoint(
            monitor="positive_residuals",
            mode="max",
            auto_insert_metric_name = True,
            filename = 'positive_residuals-{epoch}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}-{turn_around:.4f}-{positive_residuals:.4f}-{negative_residuals:.4f}-{mape:.4f}'

        )
    CheckpointCallback_mape = ModelCheckpoint(
            monitor="mape",
            mode="min",
            auto_insert_metric_name = True,
            filename = 'MAPE-{epoch}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}-{turn_around:.4f}-{positive_residuals:.4f}-{negative_residuals:.4f}-{mape:.4f}'
        )

    early_stopping_callback = EarlyStopping(monitor="turn_around", mode="max" , patience = 30)

    # hyperparameters = {
    #         "batch_size" : 128,
    #         "window_size_input" : trial.suggest_int("window_size_input" , 5, 5, log = False),
    #         "window_size_output" : 1,
    #         "input_size" : trial.suggest_categorical("input_size" , [1,5]),
    #         "hidden_layer_size" : trial.suggest_int("hidden_layer_size" , 32, 256),
    #         "num_layers" : trial.suggest_categorical("num_layers" , [1,2,3]),
    #         "output_size" : 1,
    #         "dropout" : trial.suggest_float("dropout" , 0.0, 0.5),
    #         "lr" : trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    #         "normalization" : trial.suggest_categorical("normalization" , [0,1]),
    #         "make_reference" : trial.suggest_categorical("make_reference" , [True,False]),
    #         "pruned" : False
    #     }


    # hyperparameters = {
    #         "batch_size" : 128,
    #         "window_size_input" : trial.suggest_int("window_size_input" , 5, 5, log = True),
    #         "max_seq_len" : 150,
    #         "window_size_output" : 1,
    #         "input_size" : trial.suggest_categorical("input_size" , [1,5]),
    #         "output_size" : 1,
    #         "batch_first" : False,
    #         "dim_val" : trial.suggest_categorical("dim_val",[32,64,128,256,512]),
    #         "n_heads" : trial.suggest_categorical("n_heads",[2,4,8,16,32]),
    #         "n_encoder_layers" : trial.suggest_int("n_encoder_layers" , 1, 3, log = True),
    #         "n_decoder_layers" : trial.suggest_int("n_decoder_layers" , 1, 7, log = True),
    #         "dropout_encoder" : trial.suggest_float("dropout_encoder", 1e-6, 0.5, log=True),
    #         "dropout_decoder" : trial.suggest_float("dropout_decoder", 1e-6, 0.5, log=True),
    #         "dropout_pos_enc" : trial.suggest_float("dropout_pos_enc", 1e-6,  1e-6, log=True),
    #         "dim_feedforward_encoder" : trial.suggest_int("dim_feedforward_encoder" , 32, 720, log = True),
    #         "dim_feedforward_decoder" : trial.suggest_int("dim_feedforward_decoder" , 32, 1024, log = True),
    #         "lr" : trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    #         "normalization" : trial.suggest_categorical("normalization" , [0,1]),
    #         "make_reference" : trial.suggest_categorical("make_reference" , [True,False]),
    #         "pruned" : False,
          
    #     }
    
    hyperparameters = {
            "batch_size" : 128,
            "window_size_input" : trial.suggest_int("window_size_input" , 5, 5, log = True),
            "window_size_output" : 1,
            "input_size" : trial.suggest_categorical("input_size" , [5,6]),
            "output_size" : 1,
            "lr" : trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "make_reference" : trial.suggest_categorical("make_reference" , [True,False]),
            "pruned" : False,
            "hidden_layer_1_size" : trial.suggest_int("hidden_layer_1_size" , 32, 512, log = True),
            "hidden_layer_2_size" : trial.suggest_int("hidden_layer_2_size" , 32, 512, log = True),

            }
    
    wandb.define_metric('val_loss', summary='min')
    # wandb.define_metric('val_loss_step', summary='min')
    wandb.define_metric('train_loss', summary='min')
    # wandb.define_metric('train_loss_step', summary='min')
    wandb.define_metric('turn_around', summary='max')
    wandb.define_metric('residuals', summary='max')
    wandb.define_metric('positive_residuals', summary='max')
    wandb.define_metric('negative_residuals', summary='max')
    wandb.define_metric('mape', summary='min')

   
    # wandb_logger.watch(model, log="all")

    wandb_logger.experiment.config.update(hyperparameters  , allow_val_change=True)

    # trainer.logger.log_hyperparams(hyperparameters)

    if(hyperparameters['input_size'] == 6):
        input_variables = ['close', 'open','high','low','volume','sentiment_score']
    else:
        input_variables = ['close', 'open','high','low','volume']
    
    target_variables = ['close']
    preprocessing_std = preprocessing.Preprocessing(
        prefix = data_name_prefix,
        data=data,
        window_size = hyperparameters['window_size_input'],
        target_window_size = hyperparameters["window_size_output"],
        step_size=STEP_SIZE,
        input_variables=input_variables,
        target_variables = target_variables,
        test_size=TEST_SIZE,
        normalize_data = 'None',
        make_reference= hyperparameters['make_reference']
    )

    preprocessing_std.create(scaling=normalization_options[0] , trading_day=True , group_by_day=True)

    training_data_std = preprocessing_std.get_train_loader(shuffle= True , batch_size = hyperparameters['batch_size'])
    testing_data_std= preprocessing_std.get_validation_loader(shuffle= False, batch_size = hyperparameters['batch_size'])
    

    trainer = pl.Trainer( callbacks = [ #, wandb_logger = wandb_logger
                                        PyTorchLightningPruningCallback_2(trial, monitor="turn_around", wandb_logger = wandb_logger ),
                                        CheckpointCallback_turn_around,
                                        CheckpointCallback_mape,
                                        CheckpointCallback_positive_residuals,
                                        CheckpointCallback_val,
                                    ],
                        logger=wandb_logger,
                        val_check_interval = 0.1,
                        num_sanity_val_steps = 0,
                        enable_checkpointing = True,
                        gradient_clip_val = 3,
                        gradient_clip_algorithm="value",
                        max_epochs = 35 )
    
    testing_trainer = pl.Trainer( 
                        # logger=wandb_logger,
                        # val_check_interval = 0.1,
                        #   num_sanity_val_steps = 0,
                        # enable_checkpointing = True,
                        # gradient_clip_val = 3,
                        # gradient_clip_algorithm="value",
                        max_epochs = 35 )
    

    # torch.cuda.empty_cache()
    # torch.clear_autocast_cache()
    # time.sleep(random.random()*10)
    # h = nvmlDeviceGetHandleByIndex(0)
    # info = nvmlDeviceGetMemoryInfo(h)
    # print("checking GPU memory")
    # while(info.free < 1100000000):
       
    #     time.sleep(random.random()*60)
    #     nvmlInit()
    #     h = nvmlDeviceGetHandleByIndex(0)
    #     info = nvmlDeviceGetMemoryInfo(h)
    
    model = LSTMModel(
         input_size= hyperparameters["input_size"] ,
         hidden_layer_size = hyperparameters["hidden_layer_size"],
         num_layers = hyperparameters["num_layers"],
         output_size = hyperparameters["output_size"],
         dropout = hyperparameters["dropout"],
         lr = hyperparameters["lr"]
     )



    # model = TimeSeriesTransformer(
    #     input_size = hyperparameters["input_size"] ,
    #     dec_seq_len = hyperparameters["window_size_output"] ,
    #     max_seq_len = hyperparameters["max_seq_len"] ,
    #     enc_seq_len= hyperparameters["window_size_input"] ,
    #     batch_first = hyperparameters["batch_first"] ,
    #     out_seq_len = hyperparameters["window_size_output"] ,
    #     dim_val = hyperparameters["dim_val"] ,
    #     n_heads =  hyperparameters["n_heads"] ,
    #     n_encoder_layers =  hyperparameters["n_encoder_layers"] ,
    #     n_decoder_layers = hyperparameters["n_decoder_layers"] ,
    #     dropout_encoder = hyperparameters["dropout_encoder"] ,
    #     dropout_decoder = hyperparameters["dropout_decoder"] ,
    #     dropout_pos_enc = hyperparameters["dropout_pos_enc"] ,
    #     dim_feedforward_encoder = hyperparameters["dim_feedforward_encoder"] ,
    #     dim_feedforward_decoder =  hyperparameters["dim_feedforward_decoder"] ,
    #     num_predicted_features = hyperparameters["output_size"] ,
    # )

    #model = MLP(
    #    input_size = hyperparameters['input_size'] * hyperparameters['window_size_input'],
    #    hidden_layer_1_size = hyperparameters['hidden_layer_1_size'],
    #    hidden_layer_2_size = hyperparameters['hidden_layer_2_size'],
    #    lr = hyperparameters['lr'],
    #    output_size = hyperparameters['output_size'],
    #)


    trainer.fit(model=model, train_dataloaders=training_data_std, val_dataloaders=testing_data_std)
    #model = MLP.load_from_checkpoint(CheckpointCallback_turn_around.best_model_path)
    model = LSTMModel.load_from_checkpoint("lightning_logs\\version_16\checkpoints\\turn_around-epoch=0-train_loss_epoch=0.0000-val_loss_epoch=0.0000-turn_around=0.6110-positive_residuals=0.4423-negative_residuals=-0.4642-mape=8.5935.ckpt")
    testing_trainer.test(model , testing_data_std)
    turnAround = testing_trainer.callback_metrics['turn_around'].item()

    plotTimeSeries(preprocessing_std , model , trainer,prefix = trial.number)

    wandb_logger.log_image(key="samples", images=[f".\\temps_MLP\\temp_{trial.number}.png"])
    wandb_logger.finalize('success')
    wandb_logger.experiment.finish()
    wandb.finish()


        

    del model
    del training_data_std
    del testing_trainer
    del preprocessing_std
    del testing_data_std

    torch.cuda.empty_cache()
    torch.clear_autocast_cache()


    return turnAround


seed = int(random.random() * 200)
print("random seed : " , seed)
pruner = optuna.pruners.HyperbandPruner( min_resource=15, max_resource= 350, reduction_factor=2)
sampler = TPESampler(n_startup_trials = 50, multivariate = True , seed = seed, group = True)

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    #study = optuna.load_study(
       #study_name="MLP-study-std-only",
       #storage=databseURL,
       #pruner = pruner,
       #sampler = sampler,
    #)
    #study.optimize(objective, n_trials = 110,n_jobs=1)




# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# study = optuna.create_study(directions = ['maximize'] , sampler = sampler , pruner= pruner,study_name="LSTM-study-std-minmax-only", storage=databseURL )
# study.optimize(objective, n_trials=8 , n_jobs=1)

# best_params = study.best_params
# print(best_params)
# print("finshed")





CheckpointCallback_turn_around = ModelCheckpoint(
     monitor="turn_around",
     mode="max",
     auto_insert_metric_name = True,
     filename = 'turn_around-{epoch}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}-{turn_around:.4f}-{positive_residuals:.4f}-{negative_residuals:.4f}-{mape:.4f}'

 )

preprocessing_std = preprocessing.Preprocessing(
         prefix = data_name_prefix,
         data=data,
         window_size = 5,
         target_window_size = 1,
         step_size=STEP_SIZE,
         input_variables=input_variables,
         target_variables = target_variables,
         test_size=TEST_SIZE,
         normalize_data = 'None',
         make_reference=True
     )

preprocessing_std.create(scaling='std' , trading_day=False , group_by_day=False)

training_data_std = preprocessing_std.get_train_loader(shuffle= True , batch_size = 128)
testing_data_std= preprocessing_std.get_validation_loader(shuffle= False, batch_size = 128)

# i, batch = next(enumerate(training_data_std))
# src, trg, trg_y ,id= batch

# print(src.size())
# print(trg.size())
# print(trg_y.size())
# print(id.size())

# print(src[0])
# print(trg[0])
# print(trg_y[0])
# print(id[0])

# model = TimeSeriesTransformer(
#         input_size = 5 ,
#         dec_seq_len = 1,
#         max_seq_len = 100,
#         enc_seq_len= 5,
#         batch_first = False,
#         out_seq_len = 1,
#         dim_val = 32,
#         n_heads = 32, 
#         n_encoder_layers = 2,
#         n_decoder_layers = 2,
#         dropout_encoder = 0.2,
#         dropout_decoder = 0.2,
#         dropout_pos_enc = 0.1,
#         dim_feedforward_encoder = 128,
#         dim_feedforward_decoder = 128,
#         num_predicted_features = 1
#     )

# model = MLP(
#     input_size= len(input_variables) * 5
# )
# # # # print(src[1])
# # # # print(trg[1])
# # # # print(trg_y[1])
# # # # print(id[])

model = LSTMModel(
     input_size= len(input_variables) ,
     hidden_layer_size = 256,
     num_layers = 3,
     output_size = 1,
     dropout = 0.055,
     lr =0.0006
 )



# # nvmlInit()
# h = nvmlDeviceGetHandleByIndex(0)
# info = nvmlDeviceGetMemoryInfo(h)
# print(f'total    : {info.total}')
# print(f'free     : {info.free}')
# print(f'used     : {info.used}')

trainer = pl.Trainer( callbacks = [
#                                     # CheckpointCallback_mape,
#                                     # CheckpointCallback_positive_residuals,
                                     CheckpointCallback_turn_around,
#                                     # CheckpointCallback_val,
#                                     # PyTorchLightningPruningCallback(trial, monitor="val_loss_epoch"),
#                                     # early_stopping_callback
                                 ],
                    
#                     # logger=wandb_logger,
                     precision = '16-mixed',
                     val_check_interval = 0.1,
#                     #   num_sanity_val_steps = 2,
                     enable_checkpointing = True,
                     gradient_clip_val = 5,
                     gradient_clip_algorithm="value",
                     max_epochs =30)


trainer.fit(model=model, train_dataloaders=training_data_std, val_dataloaders=testing_data_std)
# model = MLP.load_from_checkpoint(CheckpointCallback_turn_around.best_model_path)
model = LSTMModel.load_from_checkpoint("lightning_logs\\version_16\checkpoints\\turn_around-epoch=0-train_loss_epoch=0.0000-val_loss_epoch=0.0000-turn_around=0.6110-positive_residuals=0.4423-negative_residuals=-0.4642-mape=8.5935.ckpt")
trainer.test(model , testing_data_std)
plotTimeSeries(preprocessing_std , model , trainer, prefix=2987451 , verbose=True)