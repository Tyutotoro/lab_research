import datetime
import os
from pathlib import Path


class folder:
    def __init__(self):      
        self.dt_now = datetime.datetime.now()

    def cre_floder(self,save_path=None, option=None):
        path = os.path.join(save_path,self.dt_now.strftime('%Y%m%d_%H%M%S') + '_'+ option) 
        parent_path = Path(path)
        image_path = parent_path.joinpath('image')
        log_path = parent_path.joinpath('log')
        os.mkdir(path)
        os.mkdir(image_path)
        os.mkdir(log_path)
        return [path, str(image_path), str(log_path)]