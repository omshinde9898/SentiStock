import pandas as pd
import os
from pathlib import Path
from src import logger

def read_csv_headlines(paths : list[str]) -> pd.DataFrame:
    """
    read csv files from given path and returns dataframe

    Args:
        paths : list containig paths to each csv files

    return : Dataframe
    """
    try:
        # empty frame to store all instances of headlines from multile csv files
        data = pd.DataFrame(columns=['headline','label'])

        for path in paths:
            logger.info(f"reading csv file from {path}")
            # read csv file into dataframe
            frame = pd.read_csv(Path(path))

            frame = frame.loc[:,['headline','label']]

            # concate original dataframe with new dataframe 
            data = pd.concat([data,frame],axis=0,ignore_index=True)
        
        # return final dataframe
        logger.info("Returned data frame from csv files")
        return data
    
    except Exception as e:
        logger.error(f'Read csv headlines failedw ith{e}')
        raise e
    
