import pandas as pd
import os
from pathlib import Path
import logging

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

            # read csv file into dataframe
            frame = pd.read_csv(Path(path))

            frame = frame.loc[:,['headline','label']]

            # concate original dataframe with new dataframe 
            data = pd.concat([data,frame],axis=0,ignore_index=True)
        
        # return final dataframe
        return data
    
    except Exception as e:
        raise e
    
