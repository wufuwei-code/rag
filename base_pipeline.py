import time
from universal_parser import UniversalParser
from universal_rag_engine import UniversalRAGEngine
from universal_dialogue_tracker import UniversalDialogueTracker
from error_handler import ErrorHandler
import logging

# 添加日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasePipeline:
    def __init__(self):
        print("初始化通用问答系统...")
        self.parser = UniversalParser()
        self.engine = UniversalRAGEngine()
        self.dialogue_tracker = UniversalDialogueTracker()
        self.error_handler = ErrorHandler()
        self.performance_log = "performance.log"
        
    def process(self, user_query, crag_data, session_id=None, feedback=None):
        """带反馈处理的问题处理流程"""
        start_time = time.time()
        try:
            # 0. 如果有反馈，先处理反馈
            if feedback:
                self.engine.process_feedback(feedback)
            
            # 1. 对话状态管理
            resolved_query = self.dialogue_tracker.resolve_references(user_query)
            
            # 2. 语义解析
            parsed = self.parser.parse(resolved_query)
            
            # 3. 构建检索查询（考虑上下文）
            context = self.dialogue_tracker.get_context_for_retrieval()
            search_query = self.parser.build_search_query(parsed, context)
            
            # 4. 检索相关文档
            relevant_docs = self.engine.retrieve(search_query, crag_data)
            
            # 5. 生成答案
            answer, confidence, generation_data = self.engine.generate_answer(resolved_query, relevant_docs)
            
            # 6. 将本轮对话加入历史
            self.dialogue_tracker.add_to_history(
                user_query=user_query,
                system_response=answer,
                parsed_info=parsed
            )
            
            # 记录性能
            latency = time.time() - start_time
            self.monitor_performance(user_query, latency)
            
            # 返回答案和生成数据（用于可能的反馈）
            return answer, confidence, generation_data
        except Exception as e:
            latency = time.time() - start_time
            self.monitor_performance(user_query, latency)
            self.error_handler.log_error(e, user_query)
            return self.handle_error(e, user_query), 0.0, {}

    def handle_error(self, error, query):
        """通用错误处理"""
        # 返回友好提示
        error_types = {
            "NameError": "处理您的查询时遇到模块未定义问题",
            "IndexError": "处理您的查询时遇到数据问题",
            "KeyError": "系统配置可能存在问题",
            "ValueError": "查询格式可能不符合要求",
            "TypeError": "数据类型处理错误"
        }
        
        error_name = type(error).__name__
        return f"抱歉，{error_types.get(error_name, '处理您的请求时出现问题')}，请尝试简化问题或换种问法"

    def monitor_performance(self, query, latency):
        """性能监控"""
        try:
            with open(self.performance_log, "a", encoding="utf-8") as f:
                f.write(f"{time.ctime()},{query},{latency:.4f}\n")
        except Exception as e:
            logger.error(f"无法记录性能日志: {str(e)}")
