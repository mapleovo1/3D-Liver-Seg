import os
import time
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="se_com_loss_training_logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"training_{timestamp}.txt")
        self.log(f"初始化日志记录器 @ {self.log_path}", print_msg=False)
    
    def log(self, message, print_msg=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        if print_msg:
            print(log_entry)

def tic_toc(start_time=None):
    return time.time() if start_time is None else time.time() - start_time