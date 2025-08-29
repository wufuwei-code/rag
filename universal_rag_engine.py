import numpy as np
from sentence_transformers import SentenceTransformer, util
from config import Config
import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import time
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerationModelLoader:
    """生成模型加载器（单例模式）"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """初始化生成模型"""
        if not hasattr(self, 'model'):  # 防止重复初始化
            self.model_loaded = False
            self.model = None
            self.tokenizer = None
            
            if Config.USE_GENERATIVE_MODEL:
                self._load_generation_model()
    
    def _load_generation_model(self):
        """加载生成模型"""
        try:
            logger.info(f"正在加载生成模型: {Config.GENERATION_MODEL_PATH}")
            
            # 检查模型路径是否存在
            if not os.path.exists(Config.GENERATION_MODEL_PATH):
                logger.error(f"模型路径不存在: {Config.GENERATION_MODEL_PATH}")
                self.model_loaded = False
                return
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.GENERATION_MODEL_PATH,
                trust_remote_code=True
            )
            
            # 设置设备
            device = "cuda" if torch.cuda.is_available() and Config.GENERATION_DEVICE == "cuda" else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.GENERATION_MODEL_PATH,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).eval()
            
            logger.info("生成模型加载完成")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"加载生成模型失败: {str(e)}")
            self.model_loaded = False
    
    def generate(self, prompt, max_tokens=Config.MAX_NEW_TOKENS, temperature=Config.GENERATION_TEMPERATURE):
        """执行文本生成"""
        if not self.model_loaded or self.model is None:
            logger.warning("生成模型未加载，无法生成文本")
            return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成参数
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # 对于小内存设备，使用低内存模式
            if not torch.cuda.is_available():
                generate_kwargs["low_memory"] = True
            
            outputs = self.model.generate(**generate_kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            return None

class UniversalRAGEngine:
    def __init__(self):
        # 使用轻量级多语言模型
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        # 答案置信度阈值
        self.confidence_threshold = 0.8
        
        # 反馈系统初始化
        self.feedback_log = Config.FEEDBACK_LOG_PATH
        self.knowledge_update_queue = []
        self.last_feedback_time = 0
        
        self.knowledge_base = self._load_knowledge_base()  # 加载知识库
        self.correction_map = {}  # 实时纠正映射
        
        # 初始化生成模型加载器
        if Config.USE_GENERATIVE_MODEL:
            self.generator = GenerationModelLoader.get_instance()
    
    def _normalize_query(self, query):
        """规范化查询字符串以提高匹配率"""
        # 移除标点符号
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', '', query)
        # 转换为小写
        query = query.lower()
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def _load_knowledge_base(self):
        """加载知识库"""
        knowledge_base = {}
        if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
            try:
                with open(Config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        knowledge_base[entry["query"]] = entry["answer"]
                logger.info(f"已加载 {len(knowledge_base)} 条知识记录")
            except Exception as e:
                logger.error(f"加载知识库失败: {str(e)}")
        return knowledge_base
    
    def _save_knowledge_base(self):
        """保存知识库"""
        try:
            with open(Config.KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
                for query, answer in self.knowledge_base.items():
                    f.write(json.dumps({"query": query, "answer": answer}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"保存知识库失败: {str(e)}")        
    
    def retrieve(self, query, crag_data):
        """通用文档检索"""
        # 获取所有文档片段
        search_results = crag_data.get("search_results", [])
        all_snippets = [doc.get("page_snippet", "") for doc in search_results]
        
        if not all_snippets:
            return []
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算所有文档嵌入
        doc_embeddings = self.embedding_model.encode(all_snippets)
        
        # 计算相似度
        similarities = util.dot_score([query_embedding], doc_embeddings)[0].numpy()
        
        # 获取最相关的文档
        top_indices = np.argsort(similarities)[-Config.TOP_K_DOCS:][::-1]
        return [{
            "snippet": all_snippets[i],
            "similarity": float(similarities[i]),
            "doc_id": search_results[i].get("doc_id", "") if i < len(search_results) else ""
        } for i in top_indices]
    
    def generate_answer(self, query, documents):
        """通用答案生成 - 优先使用纠正后的答案"""
        generation_data = {
            "query": query,
            "documents": [],
            "answer": "",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. 检查实时纠正映射（最高优先级）
        normalized_query = self._normalize_query(query)
        if normalized_query in self.correction_map:
            answer = self.correction_map[normalized_query]
            generation_data.update({
                "answer": answer,
                "confidence": 0.99,
                "method": "correction_cache"
            })
            return answer, 0.99, generation_data
        
        # 2. 检查知识库（次高优先级）
        if normalized_query in self.knowledge_base:
            answer = self.knowledge_base[normalized_query]
            generation_data.update({
                "answer": answer,
                "confidence": 0.95,
                "method": "knowledge_base"
            })
            return answer, 0.95, generation_data
        
        # 3. 原有生成逻辑（最低优先级）
        if not documents:
            answer = self.get_unknown_answer(query)
            generation_data["answer"] = answer
            return answer, 0.0, generation_data
        
        # 记录文档信息
        for doc in documents:
            generation_data["documents"].append({
                "snippet": doc["snippet"][:100] + "..." if len(doc["snippet"]) > 100 else doc["snippet"],
                "similarity": doc["similarity"],
                "doc_id": doc.get("doc_id", "")
            })
        
        # 尝试生成式方法（如果启用）
        if Config.USE_GENERATIVE_MODEL and hasattr(self, 'generator') and self.generator.model_loaded:
            try:
                # 构建上下文
                context = self._build_context(documents)
                
                # 创建Qwen优化的提示模板
                prompt = self._create_qwen_prompt(context, query)
                
                # 生成完整响应
                full_response = self.generator.generate(prompt)
                
                if full_response:
                    # 提取答案部分
                    answer = self._extract_answer(full_response, prompt)
                    
                    # 计算置信度
                    confidence = self._calculate_confidence(query, documents, answer)
                    
                    # 如果置信度足够高，直接返回
                    if confidence >= self.confidence_threshold * 0.8:
                        generation_data["answer"] = answer
                        generation_data["confidence"] = confidence
                        generation_data["method"] = "generative"
                        return answer, confidence, generation_data
            except Exception as e:
                logger.error(f"生成式模型调用失败: {str(e)}")
                # 生成式方法失败时，继续执行下面的规则方法
        
        # 如果生成式方法未启用或失败，回退到基于规则的方法
        answer, confidence = self.rule_based_generation(query, documents)
        generation_data["answer"] = answer
        generation_data["confidence"] = confidence
        generation_data["method"] = "rule_based"
        return answer, confidence, generation_data
    
    def rule_based_generation(self, query, documents):
        """基于规则的答案生成 - 带置信度评估"""
        best_answer = ""
        best_confidence = 0.0
        
        # 尝试多种答案提取策略
        for doc in documents:
            snippet = doc["snippet"]
            similarity = doc["similarity"]
            
            # 1. 尝试提取明确答案
            if "答案：" in snippet:
                answer_candidate = snippet.split("答案：")[1].split("。")[0]
                confidence = self.calculate_confidence(query, answer_candidate, similarity)
                if confidence > best_confidence:
                    best_answer = answer_candidate
                    best_confidence = confidence
            
            # 2. 尝试提取数字答案
            if "多少" in query or "几个" in query:
                numbers = re.findall(r'\d[\d,\.亿万千]*', snippet)
                if numbers:
                    answer_candidate = f"大约 {numbers[0]}"
                    confidence = self.calculate_confidence(query, answer_candidate, similarity)
                    if confidence > best_confidence:
                        best_answer = answer_candidate
                        best_confidence = confidence
            
            # 3. 尝试提取日期答案
            if "日期" in query or "时间" in query:
                dates = re.findall(r'\d{4}年?\d{1,2}月?\d{1,2}日?', snippet)
                if dates:
                    answer_candidate = dates[0]
                    confidence = self.calculate_confidence(query, answer_candidate, similarity)
                    if confidence > best_confidence:
                        best_answer = answer_candidate
                        best_confidence = confidence
        
        # 4. 尝试返回最相关句子
        if best_confidence < self.confidence_threshold:
            first_sentence = documents[0]["snippet"].split("。")[0] + "。"
            answer_candidate = first_sentence if len(first_sentence) > 5 else documents[0]["snippet"][:100] + "..."
            confidence = self.calculate_confidence(query, answer_candidate, documents[0]["similarity"])
            if confidence > best_confidence:
                best_answer = answer_candidate
                best_confidence = confidence
        
        # 5. 置信度不足时返回未知提示
        if best_confidence < self.confidence_threshold:
            return self.get_unknown_answer(query), best_confidence
        
        return best_answer, best_confidence
    
    def _build_context(self, documents):
        """构建检索上下文"""
        if not documents:
            return "无相关信息"
        
        context_parts = []
        for i, doc in enumerate(documents[:Config.TOP_K_CONTEXT]):  # 使用TOP_K_CONTEXT个文档
            context_parts.append(f"### 参考文档 {i+1} (相关度: {doc['similarity']:.2f}) ###\n{doc['snippet']}\n")
        return "\n".join(context_parts)
    
    def _create_qwen_prompt(self, context, query):
        """创建Qwen优化的提示模板"""
        return f"""
        <|system|>
        你是一个专业问答助手，根据用户问题提供准确、简洁的回答。
        请仅使用提供的文档信息回答问题，不要编造信息。
        
        提供的参考信息如下：
        {context}
        </s>
        
        <|user|>
        {query}
        </s>
        
        <|assistant|>
        """
    
    def _extract_answer(self, full_response, prompt):
        """从模型响应中提取答案"""
        # 移除提示部分，只保留新生成的内容
        answer = full_response[len(prompt):].strip()
        
        # 清理可能的结束标记
        for end_marker in ["</s>", "<|endoftext|>", "###", "<|im_end|>"]:
            if end_marker in answer:
                answer = answer.split(end_marker)[0]
        
        # 移除多余的空白和换行
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # 如果以标点符号结尾，保留原样；否则添加句号
        if answer and answer[-1] not in ['.', '!', '?', '。', '！', '？']:
            answer += '。'
            
        return answer
    
    def _calculate_confidence(self, query, documents, answer):
        """计算答案置信度（更精确的版本）"""
        if not documents:
            return 0.0
        
        # 基础置信度 = 最佳文档相似度
        confidence = documents[0]["similarity"] * 0.5
        
        # 答案长度加分
        if len(answer) > 15:
            confidence += 0.1
        
        # 答案完整性加分（包含句号表示完整句子）
        if any(punct in answer for punct in ['.', '。', '!', '！']):
            confidence += 0.1
        
        # 文档支持加分
        for doc in documents[:2]:
            if doc["snippet"] in answer:
                confidence += 0.1
                break
        
        # 关键词匹配加分
        keywords = set(self._parse_query(query)["keywords"])
        keyword_count = sum(1 for keyword in keywords if keyword in answer)
        confidence += min(keyword_count * 0.05, 0.2)  # 每个关键词最多加0.05，上限0.2
        
        return min(max(confidence, 0.0), 0.95)  # 保留5%不确定性
    
    def _parse_query(self, query):
        """基础语义解析（简化实现）"""
        # 实际应用中应使用更复杂的解析器
        return {"keywords": [word for word in query.split() if len(word) > 1]}
    
    def calculate_confidence(self, query, answer, similarity):
        """计算答案置信度（原始版本）"""
        # 基础置信度 = 文档相似度
        confidence = similarity * 0.7
        
        # 答案长度加分
        if len(answer) > 5:
            confidence += 0.1
        
        # 关键词匹配加分
        keywords = set(self._parse_query(query)["keywords"])
        for keyword in keywords:
            if keyword in answer:
                confidence += 0.05
                break
        
        # 确保置信度在合理范围内
        return min(max(confidence, 0.0), 1.0)
    
    def get_unknown_answer(self, query):
        """未知问题的回答"""
        unknown_responses = [
            "这个问题超出了我的知识范围",
            "我还没有学习过这方面的知识",
            "暂时无法回答这个问题，建议咨询相关专家",
            "这个问题很有趣，但我目前无法提供准确答案"
        ]
        
        # 根据问题类型定制回答
        if "怎么" in query or "如何" in query:
            return "我还没有掌握解决这个问题的方法"
        if "为什么" in query:
            return "这个问题的原因我还不太清楚"
        
        # 随机选择一种回答方式
        import random
        return random.choice(unknown_responses)
    
    def process_feedback(self, feedback_data):
        """
        处理用户反馈
        Args:
            feedback_data (dict): 包含反馈信息的字典，格式为:
                {
                    "type": "negative" | "correction" | "positive",
                    "generation_data": 原始生成数据,
                    "comment": "用户评论",
                    "correction": "用户提供的正确答案（仅当type为correction时）"
                }
        """
        try:
            feedback_type = feedback_data.get("type", "negative")
            generation_data = feedback_data.get("generation_data", {})
            user_comment = feedback_data.get("comment", "")
            correction = feedback_data.get("correction", "")
            
            # 记录反馈日志
            self._log_feedback(feedback_type, generation_data, user_comment, correction)
            
            # 处理不同类型的反馈
            if feedback_type == "correction":
                self._handle_correction(generation_data, correction)
            elif feedback_type == "negative":
                self._handle_negative_feedback(generation_data, user_comment)
            elif feedback_type == "positive":
                self._handle_positive_feedback(generation_data)
            
            # 定期处理知识更新队列
            current_time = time.time()
            if current_time - self.last_feedback_time > 3600:  # 每小时处理一次
                self._process_knowledge_updates()
                self.last_feedback_time = current_time
                
            return True
        except Exception as e:
            logger.error(f"处理反馈失败: {str(e)}")
            return False
    
    def _log_feedback(self, feedback_type, generation_data, comment, correction=""):
        """记录反馈到日志文件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "query": generation_data.get("query", ""),
            "original_answer": generation_data.get("answer", ""),
            "confidence": generation_data.get("confidence", 0.0),
            "method": generation_data.get("method", ""),
            "comment": comment,
            "correction": correction
        }
        
        try:
            with open(self.feedback_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logger.info(f"反馈已记录: {feedback_type} - {generation_data.get('query', '')}")
        except Exception as e:
            logger.error(f"无法写入反馈日志: {str(e)}")
    
    def _handle_correction(self, generation_data, correction):
        """处理用户更正 - 增强匹配能力"""
        query = generation_data.get("query", "")
        if not query or not correction:
            return
            
        # 规范化查询
        normalized_query = self._normalize_query(query)
        
        # 1. 立即应用到纠正映射
        self.correction_map[normalized_query] = correction
        
        # 2. 添加到知识库
        self.knowledge_base[normalized_query] = correction
        self._save_knowledge_base()  # 持久化保存
        
        # 3. 添加到知识更新队列
        self._add_to_knowledge_update_queue(query, correction)
        
        logger.info(f"已应用实时更新: 问题『{query}』(规范化: {normalized_query})的答案更新为『{correction}』")
    
    def _add_to_knowledge_update_queue(self, query, correction):
        """添加到知识更新队列"""
        self.knowledge_update_queue.append({
            "query": query,
            "correction": correction,
            "timestamp": datetime.now().isoformat()
        })
    
    def _process_knowledge_updates(self):
        """处理知识更新队列 - 增强实时更新"""
        if not self.knowledge_update_queue:
            return
            
        logger.info(f"开始处理 {len(self.knowledge_update_queue)} 个知识更新请求")
        
        # 实际系统中这里会更新向量数据库
        try:
            with open(Config.KNOWLEDGE_UPDATE_PATH, "a", encoding="utf-8") as f:
                for update in self.knowledge_update_queue:
                    f.write(json.dumps(update, ensure_ascii=False) + "\n")
            
            logger.info(f"知识更新请求已保存到 {Config.KNOWLEDGE_UPDATE_PATH}")
            self.knowledge_update_queue = []  # 清空队列
        except Exception as e:
            logger.error(f"保存知识更新失败: {str(e)}")    
    
    def _handle_negative_feedback(self, generation_data, comment):
        """处理负面反馈"""
        # 调整置信度阈值
        original_confidence = generation_data.get("confidence", 0)
        if original_confidence > self.confidence_threshold:
            # 提高阈值以避免类似错误
            self.confidence_threshold = min(0.8, self.confidence_threshold + 0.05)
            logger.info(f"提高置信度阈值至: {self.confidence_threshold}")
        
        # 分析失败原因
        if "不相关" in comment or "无关" in comment:
            self._analyze_retrieval_failure(generation_data)
        elif "过时" in comment or "旧" in comment:
            self._flag_outdated_documents(generation_data.get("documents", []))
        elif "错误" in comment or "不正确" in comment:
            self._analyze_factual_error(generation_data)
            
    
    def _handle_positive_feedback(self, generation_data):
        """处理正面反馈"""
        # 强化成功模式
        original_confidence = generation_data.get("confidence", 0)
        if original_confidence < self.confidence_threshold:
            # 降低阈值以接受类似答案
            self.confidence_threshold = max(0.1, self.confidence_threshold - 0.03)
            logger.info(f"降低置信度阈值至: {self.confidence_threshold}")
    
    def _analyze_retrieval_failure(self, generation_data):
        """分析检索失败原因"""
        query = generation_data.get("query", "")
        if not query:
            return
            
        logger.info(f"分析检索失败: 查询『{query}』")
        
        # 实际系统中这里会进行更深入的分析，例如：
        # 1. 检查查询与文档的语义差距
        # 2. 评估嵌入模型在该领域的表现
        # 3. 调整检索参数
        
        # 临时解决方案：记录问题供后续分析
        with open("retrieval_issues.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {query}\n")
    
    def _flag_outdated_documents(self, documents):
        """标记过时文档"""
        if not documents:
            return
            
        outdated_docs = []
        for doc in documents:
            doc_id = doc.get("doc_id", "")
            if doc_id:
                outdated_docs.append(doc_id)
                logger.info(f"标记可能过时的文档: {doc_id}")
        
        # 保存到文件供内容团队处理
        if outdated_docs:
            with open("outdated_documents.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - {', '.join(outdated_docs)}\n")
    
    def _analyze_factual_error(self, generation_data):
        """分析事实错误"""
        query = generation_data.get("query", "")
        answer = generation_data.get("answer", "")
        
        if not query or not answer:
            return
            
        logger.info(f"分析事实错误: 问题『{query}』，回答『{answer}』")
        
        # 实际系统中这里会：
        # 1. 检查答案与来源文档的一致性
        # 2. 评估生成模型的可靠性
        # 3. 识别可能的幻觉问题
        
        # 临时解决方案：记录问题供后续分析
        with open("factual_errors.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - Q: {query} | A: {answer}\n")
