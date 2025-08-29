import logging
import json
import traceback
from datetime import datetime

class ErrorHandler:
    def __init__(self, log_file="system_errors.log"):
        self.log_file = log_file
        logging.basicConfig(filename=log_file, level=logging.ERROR, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def log_error(self, error, query):
        """记录错误日志"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc()
        }
        
        # 写入日志文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_info, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"无法写入错误日志: {str(e)}")
        
        # 记录到标准错误日志
        logging.error(json.dumps(error_info, ensure_ascii=False))
