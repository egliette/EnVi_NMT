import os
import time 


class TimeContextManager():
    def __enter__(self):
        self.start_time = time.time()
        return self

    def get_time(self):
        elapsed_time = time.time() - self.start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def __exit__(self, *args):
        pass
        
def exist(path):
    return os.path.exists(path)
    
def create_dir(dpath):
    if not exist(dpath):
        os.makedirs(dpath)