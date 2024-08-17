from src.pipelines import TrainingPipeline , EvaluationPipeline
from src.utils import read_yaml_config
import sys
    

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
            pipeline = TrainingPipeline()
            try:
                pipeline.run_pipeline()
            except Exception as e:
                raise e
     
        if sys.argv[1] == 'test':
            model_config = read_yaml_config('config/model_config.yaml')
            pipeline = EvaluationPipeline()
            try:
                pipeline.run_pipeline(
                    log_path=model_config['LOG_PATH']
                )
            except Exception as e:
                raise e
     
        if sys.argv[1] == 'new':
            model_config = read_yaml_config('config/model_config.yaml')
            train_pipeline = TrainingPipeline()
            test_pipeline = EvaluationPipeline()
            try:
                train_pipeline.run_pipeline()
                test_pipeline.run_pipeline(
                    log_path=model_config['LOG_PATH']
                )
            except Exception as e:
                raise e
        
    else:
        print_help()