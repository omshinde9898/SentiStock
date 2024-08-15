from src.pipelines import TrainingPipeline , EvaluationPipeline
from src.utils import read_yaml_config

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
    

if __name__ == "__main__":
    run_evaluation(
        config_path={
            'db_config':'config/database_config.yaml',
            'model_config' : 'config/model_config.yaml'
        }
    )