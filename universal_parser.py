import jieba
import re
from config import Config
import jieba.posseg as pseg  # 添加缺失的导入

class UniversalParser:
    def __init__(self):
        # 加载领域词典
        try:
            jieba.load_userdict(Config.DOMAIN_DICT_PATH)
        except FileNotFoundError:
            print(f"警告: 领域词典文件 {Config.DOMAIN_DICT_PATH} 不存在，将使用默认分词")
    
    def parse(self, query):
        """通用解析方法"""
        return {
            "raw_query": query,
            "keywords": self.extract_keywords(query),
            "entities": self.extract_entities(query),  # 新增实体提取
            "intent": self.detect_intent(query)        # 新增意图识别
        }
    
    def extract_keywords(self, query):
        """提取关键词"""
        words = jieba.lcut(query)
        # 过滤停用词
        return [word for word in words 
                if word not in Config.STOP_WORDS 
                and len(word) > 1
                and not re.match(r'\W+', word)]
    
    def extract_entities(self, query):
        """提取实体"""
        words = pseg.cut(query)  # 使用导入的 pseg
        return [word for word, flag in words 
                if flag.startswith('n') 
                and word not in Config.STOP_WORDS
                and len(word) > 1]
    
    def detect_intent(self, query):
        """识别问题意图"""
        if "如何" in query or "怎样" in query:
            return "method"
        elif "为什么" in query:
            return "reason"
        elif "多少" in query or "几" in query:
            return "quantity"
        elif "何时" in query or "时间" in query:
            return "time"
        elif "哪里" in query or "地点" in query:
            return "location"
        return "general"
    
    def build_search_query(self, parsed, context=""):
        """构建检索查询（考虑上下文）"""
        # 基础关键词
        keywords = parsed["keywords"][:5]
        
        # 添加上下文实体
        if context:
            context_entities = self.extract_entities(context)
            keywords.extend(context_entities[:2])
        
        # 去重
        keywords = list(set(keywords))
        return " ".join(keywords)
