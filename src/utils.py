import os
import sys

import numpy as np
import pandas as pd
import dill #help to create a pickle file

from src.exception import customexception

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise customexception(e,sys)