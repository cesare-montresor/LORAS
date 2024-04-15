from loras.utils import load_args_and_config
from ax.service.managed_loop import optimize

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from loras.model import LORAS 
from loras.datasets import Assembly101
from loras.utils import load_args_and_config


def main():
    params = [
       
        {"name": "accumulate_grad_batches",     "value_type": "int",    "type": "range", "bounds": [2, 10]}, # 4
        {"name": "learning_rate",               "value_type": "float",  "type": "range", "bounds": [1e-6, 0.1], "log_scale": True}, # 0.01
        {"name": "weight_decay",                "value_type": "float",  "type": "range", "bounds": [0.0001, 0.001]}, # 0.0005
        {"name": "scheduler_step",              "value_type": "int",    "type": "range", "bounds": [50, 100]}, # 100
        #{"name": "batch_size",                  "value_type": "int",    "type": "range", "bounds": [4, 128]}, # 1
        #{"name": "test_batch_size",             "value_type": "int",    "type": "range", "bounds": [4, 128]}, # 1
        {"name": "cemse_alpha",                 "value_type": "float",  "type": "range", "bounds": [0.1, 0.9]}, # 0.17
        {"name": "train_epochs",                "value_type": "int",    "type": "range", "bounds": [100, 400]}, # 200
        #{"name": "frame_features",               "value_type": "int",    "type": "range", "bounds": [4, 128]}, # 2048
        #{"name": "pose_joint_features",          "value_type": "int",    "type": "range", "bounds": [4, 128]}, # 3
        #{"name": "pose_joint_count",             "value_type": "int",    "type": "range", "bounds": [4, 128]}, # 42
        {"name": "model_dim",                   "value_type": "int",    "type": "range", "bounds": [64, 512]}, # 256
        {"name": "dropout",                     "value_type": "float",  "type": "range", "bounds": [0.1, 0.9]}, # 0.20
        {"name": "temporal_layers_count",       "value_type": "int",    "type": "range", "bounds": [1, 10]}, # 6
        {"name": "temporal_state_dim",          "value_type": "int",    "type": "range", "bounds": [128, 512]}, # 512
    ]

    
    best_params, stats, experiment, model = optimize(
        parameters=params,
        evaluation_function=train_evaluate,
        objective_name='val/loss_total',
        minimize=True
    )

    print(best_params)
    print(stats)


def train_evaluate(params):
    config = load_args_and_config()
    print(config)

    config.accumulate_grad_batches = params.get("accumulate_grad_batches",4)
    config.learning_rate = params.get("learning_rate",0.01) # 0.01
    config.weight_decay = params.get("weight_decay",0.0005) # 0.0005
    config.scheduler_step = params.get("scheduler_step",100) # 100
    #config.batch_size = params.get("batch_size",1) # 1
    #config.test_batch_size = params.get("test_batch_size",1) # 1
    config.cemse_alpha = params.get("cemse_alpha", 0.17) # 0.17
    config.train_epochs = params.get("train_epochs", 200) # 200
    #config.frame_features = params.get("frame_features", 2048) # 2048
    #config.pose_joint_features = params.get("pose_joint_features", 3) # 3
    #config.pose_joint_count = params.get("pose_joint_count", 42) # 42
    config.model_dim = params.get("model_dim", 256) # 256
    config.dropout = params.get("dropout", 0.20) # 0.20
    config.temporal_layers_count = params.get("temporal_layers_count", 6) # 6
    config.temporal_state_dim = params.get("temporal_state_dim", 512) # 512


    dataset = Assembly101(config)
    model = LORAS(config)

    # Stop training if the validation loss doesn't decrease
    early_stopping = EarlyStopping(monitor='val/loss_total', patience=20, mode='min')

    logger = WandbLogger(name='LORAS', save_dir='runs')
    trainer = Trainer(
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=1 if True else config.train_epochs,
        callbacks=[early_stopping], 
        logger=logger
    )

    trainer.fit(
        model, 
        train_dataloaders=dataset.train_dataloader(), 
        val_dataloaders=dataset.val_dataloader()
    )
    #metrics = trainer.callback_metrics
    #val_loss = metrics['val/loss_total']
    #print(metrics)
    #return val_loss

    # Final evaluation to use with facebook ax
    predictions = trainer.predict(model, dataloaders=dataset.val_dataloader())
    final_score = sum(predictions) / len(predictions)
    print(f'model score: {final_score}')
    
    return {'val/loss_total':final_score}



if __name__ == '__main__':
    main()
