from src.pipelines import TrainingPipeline , EvaluationPipeline
from src.utils import read_yaml_config
import sys

def run_training(config_path:dict):
    db_config = read_yaml_config(config_path['db_config'])
    model_path = read_yaml_config(config_path['model_config'])
    pipeline = TrainingPipeline(
        db_config={
        "host" : db_config['HOST'],
        "user" : db_config['USER'],
        "password" : db_config['PASSWORD'],
        "database" : db_config['DATABASE']
    },
        model_filepath=model_path['FILEPATH']
    )
    try:
        pipeline.run_pipeline()
    except Exception as e:
        raise e
    
def run_evaluation(config_path:dict):
    db_config = read_yaml_config(config_path['db_config'])
    model_config = read_yaml_config(config_path['model_config'])
    pipeline = EvaluationPipeline(
        db_config={
        "host" : db_config['HOST'],
        "user" : db_config['USER'],
        "password" : db_config['PASSWORD'],
        "database" : db_config['DATABASE']
    },
        model_filepath=model_config['FILEPATH']
    )
    try:
        pipeline.run_pipeline(
            log_path=model_config['LOG_PATH']
        )
    except Exception as e:
        raise e
    

def run_train_eval(config_path:dict) -> None:
    run_training(
        config_path=config_path
    )
    run_evaluation(
        config_path=config_path
    )
    

def print_help():
    print('--------------------------------------------------------------------------------------')
    print('Welcome To SentiStock')
    print('Before running script make sure you change configurations in `config/` folder')
    print('Run models with arguments as follow')
    print('`python main.py train`')
    print('`python main.py test`')
    print('`python main.py new`')
    print('use new for running experiment with untrained model ')
    print('--------------------------------------------------------------------------------------')

if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        
        if sys.argv[1] == 'train':
            run_training(
                config_path={
                    'db_config':'config/database_config.yaml',
                    'model_config' : 'config/model_config.yaml'
                }
            )
     
        if sys.argv[1] == 'test':
            run_evaluation(
                config_path={
                    'db_config':'config/database_config.yaml',
                    'model_config' : 'config/model_config.yaml'
                }
            )
     
        if sys.argv[1] == 'new':
            run_train_eval(
                config_path={
                    'db_config':'config/database_config.yaml',
                    'model_config' : 'config/model_config.yaml'
                }
            )
        
    else:
        print_help()