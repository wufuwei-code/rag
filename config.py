import logging
import json
import traceback
from datetime import datetime
import torch

class Config:
    # 系统配置
    EMBEDDING_MODEL = "models/paraphrase-multilingual-MiniLM-L12-v2"  # 更小的嵌入模型
    TOP_K_DOCS = 3
    TOP_K_CONTEXT = 1  # 仅使用最相关的文档
    PERFORMANCE_LOG = "performance.log"
    ERROR_LOG = "system_errors.log"
    
    # CRAG数据路径
    CRAG_DATA_PATH = "data/crag_task_1_dev_v4.jsonl"
    
    # 停用词列表
    STOP_WORDS = [
        "的", "了", "吗", "呢", "是", "什么", "如何", "怎样", 
        "请问", "能不能", "可以", "告诉", "一下", "我"
    ]
    
    # 领域词典路径
    DOMAIN_DICT_PATH = "data/domain_dict.txt"
    
    # 生成模型配置 - 使用更小的1.8B模型
    USE_GENERATIVE_MODEL = True
    GENERATION_MODEL_PATH = "models/Qwen1.5-1.8B-Chat"  # 更小的1.8B模型
    GENERATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测
    MAX_NEW_TOKENS = 256  # 减少生成长度
    GENERATION_TEMPERATURE = 0.6
    LOAD_IN_4BIT = True 
    
    CONTEXT_WINDOW_SIZE = 5  # 考虑的对话轮次
    SUMMARY_INTERVAL = 5     # 生成摘要的对话间隔
    
    # 反馈系统配置
    FEEDBACK_LOG_PATH = "feedback.log"
    KNOWLEDGE_UPDATE_PATH = "knowledge_updates.json"
    KNOWLEDGE_BASE_PATH = "knowledge_base.json"  # 新增知识库路径
    
    # 新增实时更新配置
    REAL_TIME_UPDATE = True
