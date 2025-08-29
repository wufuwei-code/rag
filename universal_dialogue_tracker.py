import jieba
import jieba.posseg as pseg
from config import Config
import re
import logging
import numpy as np
from collections import defaultdict, deque

# 设置日志
logger = logging.getLogger(__name__)

class UniversalDialogueTracker:
    def __init__(self):
        self.history = []
        self.context_summary = ""
        self.entity_graph = {}
        self.current_focus = None
        self.focus_history = deque(maxlen=5)  # 焦点历史记录
        self.entity_mentions = defaultdict(list)  # 实体提及记录
        
        # 扩展代词映射
        self.pronoun_map = {
            # 单数代词
            "它": {"type": "singular", "gender": "neutral", "formality": "informal"},
            "这个": {"type": "singular", "gender": "neutral", "formality": "informal"},
            "那个": {"type": "singular", "gender": "neutral", "formality": "informal"},
            "他": {"type": "singular", "gender": "male", "formality": "informal"},
            "她": {"type": "singular", "gender": "female", "formality": "informal"},
            "该": {"type": "singular", "gender": "neutral", "formality": "formal"},
            "此": {"type": "singular", "gender": "neutral", "formality": "formal"},
            
            # 复数代词
            "它们": {"type": "plural", "gender": "neutral", "formality": "informal"},
            "这些": {"type": "plural", "gender": "neutral", "formality": "informal"},
            "那些": {"type": "plural", "gender": "neutral", "formability": "informal"},
            "他们": {"type": "plural", "gender": "male", "formality": "informal"},
            "她们": {"type": "plural", "gender": "female", "formality": "informal"},
            
            # 所有格代词
            "它的": {"type": "possessive", "gender": "neutral", "formality": "informal"},
            "他的": {"type": "possessive", "gender": "male", "formality": "informal"},
            "她的": {"type": "possessive", "gender": "female", "formality": "informal"},
            "它们的": {"type": "possessive", "gender": "neutral", "formality": "informal"},
            "他们的": {"type": "possessive", "gender": "male", "formality": "informal"},
            "她们的": {"type": "possessive", "gender": "female", "formability": "informal"},
        }
        
        # 扩展省略词映射
        self.ellipsis_map = {
            "呢": {"patterns": ["呢", "呢?", "呢？"], "context_required": True},
            "怎么样": {"patterns": ["怎么样", "怎么样?", "怎么样？"], "context_required": True},
            "如何": {"patterns": ["如何", "如何?", "如何？"], "context_required": True},
            "什么": {"patterns": ["什么", "什么?", "什么？"], "context_required": True},
        }
        
        # 实体类型映射
        self.entity_type_map = {
            'nr': 'person', 'nz': 'organization', 'ns': 'location',
            'nt': 'organization', 'n': 'common'
        }
        
        # 比较词列表 - 新增
        self.comparison_words = ["相比", "比较", "对比", "比起", "相较于", "和", "与", "跟"]
        
        try:
            jieba.load_userdict(Config.DOMAIN_DICT_PATH)
        except FileNotFoundError:
            print(f"警告: 领域词典文件 {Config.DOMAIN_DICT_PATH} 不存在，将使用默认分词")
    
    def resolve_references(self, query):
        """增强指代消解，使用更先进的算法"""
        logger.info(f"开始指代消解: {query}")
        
        if not self.history:
            self.history.append({"user_query": query})
            entities = self.get_entities_from_query(query)
            if entities:
                self.current_focus = entities[0]
                self.focus_history.append(entities[0])
            return query
        
        # 1. 检测并处理所有代词
        resolved_query = self._resolve_pronouns_with_attention(query)
        
        # 2. 检查是否为比较问题，如果是则跳过省略处理
        if not self._is_comparison_query(resolved_query):
            # 处理省略问法
            resolved_query = self._resolve_ellipsis_with_context(resolved_query)
        
        # 3. 更新上下文摘要
        self._update_context_summary_advanced(query)
        
        # 4. 更新实体关系图和焦点
        self._update_entity_graph_with_coreference(query)
        
        logger.info(f"指代消解完成: {resolved_query}")
        return resolved_query
    
    def _is_comparison_query(self, query):
        """检查是否为比较问题"""
        # 检查是否包含比较词
        for word in self.comparison_words:
            if word in query:
                return True
        
        # 检查是否包含"比"字
        if "比" in query:
            return True
            
        return False
    
    def _resolve_pronouns_with_attention(self, query):
        """使用注意力机制处理代词指代"""
        words = list(pseg.cut(query))
        resolved_words = []
        
        for i, (word, flag) in enumerate(words):
            if word in self.pronoun_map:
                # 找到匹配的代词
                pronoun_info = self.pronoun_map[word]
                antecedent = self._find_antecedent_with_attention(pronoun_info, word, words, i)
                
                if antecedent:
                    logger.info(f"将代词 '{word}' 替换为 '{antecedent}'")
                    resolved_words.append(antecedent)
                    # 更新当前焦点
                    self.current_focus = antecedent
                    self.focus_history.append(antecedent)
                    continue
            
            resolved_words.append(word)
        
        return "".join(resolved_words)
    
    def _find_antecedent_with_attention(self, pronoun_info, pronoun, current_words, position):
        """使用注意力评分机制寻找最合适的前指对象"""
        pronoun_type = pronoun_info["type"]
        pronoun_gender = pronoun_info["gender"]
        
        # 获取历史中的所有实体及其上下文
        all_entities = self._get_all_entities_with_context()
        
        if not all_entities:
            return None
        
        # 计算每个实体的注意力分数
        candidates = []
        for entity_info in all_entities:
            entity = entity_info["entity"]
            entity_type = entity_info["type"]
            turn_index = entity_info["turn_index"]
            context = entity_info["context"]
            
            # 计算实体与代词的匹配分数
            score = self._calculate_attention_score(
                pronoun_info, entity, entity_type, turn_index, context, current_words
            )
            
            if score > 0:  # 只有分数大于0的才考虑
                candidates.append((entity, score))
        
        # 如果没有候选，返回None
        if not candidates:
            return None
        
        # 按匹配分数排序，选择最高分的候选
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_attention_score(self, pronoun_info, entity, entity_type, turn_index, context, current_words):
        """计算实体与代词的注意力分数"""
        score = 0
        
        # 1. 时间距离：越近的提及分数越高（指数衰减）
        recency_score = np.exp(-0.5 * (len(self.history) - turn_index))
        score += recency_score * 0.4
        
        # 2. 语法角色匹配
        grammatical_score = self._check_grammatical_match(pronoun_info, entity_type)
        score += grammatical_score * 0.2
        
        # 3. 语义一致性
        semantic_score = self._check_semantic_consistency(entity, context, current_words)
        score += semantic_score * 0.2
        
        # 4. 焦点连续性
        if entity == self.current_focus:
            score += 0.15
        
        # 5. 提及频率
        mention_count = len(self.entity_mentions.get(entity, []))
        frequency_score = min(mention_count * 0.05, 0.1)  # 最多0.1分
        score += frequency_score
        
        # 6. 句法位置（如果是主语，得分更高）
        if self._is_likely_subject(entity, context):
            score += 0.05
        
        return score
    
    def _check_grammatical_match(self, pronoun_info, entity_type):
        """检查语法一致性"""
        # 单复数一致性
        if pronoun_info["type"] == "singular" and entity_type in ["person", "common"]:
            return 0.2
        elif pronoun_info["type"] == "plural" and entity_type in ["organization"]:
            return 0.2
        
        # 性别一致性（中文中较难判断，简化处理）
        if pronoun_info["gender"] == "neutral":
            return 0.1
        
        return 0
    
    def _check_semantic_consistency(self, entity, context, current_words):
        """检查语义一致性"""
        # 简单的语义一致性检查
        current_text = "".join([word for word, flag in current_words])
        
        # 如果当前上下文与实体历史上下文有重叠词汇，增加分数
        context_words = set(jieba.lcut(context))
        current_words_set = set(jieba.lcut(current_text))
        
        overlap = len(context_words.intersection(current_words_set))
        return min(overlap * 0.05, 0.2)  # 最多0.2分
    
    def _is_likely_subject(self, entity, context):
        """判断实体在上下文中是否可能是主语"""
        # 简单实现：检查实体是否出现在句首附近
        context_sentences = re.split(r'[。！？.!?]', context)
        for sentence in context_sentences:
            if entity in sentence and sentence.find(entity) < min(5, len(sentence)//2):
                return True
        return False
    
    def _get_all_entities_with_context(self):
        """从历史中获取所有实体及其上下文信息"""
        all_entities = []
        
        # 从最近到最早遍历历史
        for i in range(len(self.history)-1, -1, -1):
            query = self.get_query_from_turn(i)
            response = self.get_response_from_turn(i) if i < len(self.history) else ""
            
            # 获取上下文（当前查询+系统响应）
            context = f"{query} {response}"
            
            # 修复：使用正确的方法名
            entities_with_types = self.get_entities_with_types(query)
            for entity, entity_type in entities_with_types:
                all_entities.append({
                    "entity": entity,
                    "type": entity_type,
                    "turn_index": i,
                    "context": context
                })
        
        return all_entities
    
    def get_entities_with_types(self, query):
        """提取实体及其类型"""
        words = pseg.cut(query)
        entities = []
        
        for word, flag in words:
            if flag.startswith('n') and word not in Config.STOP_WORDS and len(word) > 1:
                entity_type = self.entity_type_map.get(flag, 'common')
                entities.append((word, entity_type))
        
        return entities
    
    def _resolve_ellipsis_with_context(self, query):
        """使用上下文处理省略问法"""
        # 检查是否为省略问法
        ellipsis_type = self._detect_ellipsis_type(query)
        if not ellipsis_type:
            return query
        
        # 如果有当前焦点，使用焦点补全
        if self.current_focus:
            return self._complete_ellipsis_with_focus(query, ellipsis_type)
        
        # 否则尝试从历史中推断
        return self._complete_ellipsis_from_history(query, ellipsis_type)
    
    def _detect_ellipsis_type(self, query):
        """检测省略类型"""
        # 如果查询中包含比较词，则不认为是省略问法
        if self._is_comparison_query(query):
            return None
            
        for ellipsis_word, info in self.ellipsis_map.items():
            for pattern in info["patterns"]:
                if pattern in query:
                    return ellipsis_word
        return None
    
    def _complete_ellipsis_with_focus(self, query, ellipsis_type):
        """使用当前焦点补全省略"""
        if ellipsis_type == "呢":
            return f"{self.current_focus}的情况如何"
        elif ellipsis_type == "怎么样":
            return f"{self.current_focus}的情况怎么样"
        elif ellipsis_type == "如何":
            return f"{self.current_focus}的情况如何"
        elif ellipsis_type == "什么":
            return f"{self.current_focus}是什么"
        
        return query
    
    def _complete_ellipsis_from_history(self, query, ellipsis_type):
        """从历史中推断省略内容"""
        if not self.history:
            return query
        
        # 获取上一轮的问题类型
        last_query = self.get_last_query()
        
        # 提取上一轮的主要实体
        last_entities = self.get_entities_from_query(last_query)
        if not last_entities:
            return query
        
        primary_entity = last_entities[0]
        
        # 根据上一轮问题类型生成完整问题
        if "如何" in last_query or "怎样" in last_query:
            return f"{primary_entity}的情况如何"
        elif "为什么" in last_query:
            return f"{primary_entity}的原因是什么"
        elif "多少" in last_query or "几" in last_query:
            return f"{primary_entity}有多少"
        elif "何时" in last_query or "时间" in last_query:
            return f"{primary_entity}的时间是什么时候"
        else:
            return f"{primary_entity}的情况怎么样"
    
    def _update_context_summary_advanced(self, query):
        """使用更先进的方法生成对话摘要"""
        # 每2轮对话生成一次摘要
        if len(self.history) % 2 == 0:
            # 获取最近3轮对话
            recent_turns = min(3, len(self.history))
            context_parts = []
            
            for i in range(-recent_turns, 0):
                user_query = self.get_query_from_turn(i)
                system_response = self.get_response_from_turn(i)
                context_parts.append(f"用户: {user_query}")
                context_parts.append(f"系统: {system_response}")
            
            context_text = "\n".join(context_parts)
            
            # 提取关键实体和话题
            key_entities = self._extract_key_entities(context_text)
            
            if key_entities:
                self.context_summary = f"对话围绕{key_entities[0]}等实体展开，涉及相关问题和解答"
            else:
                # 使用关键词生成摘要
                words = jieba.lcut(context_text)
                keywords = [word for word in words if len(word) > 1 and word not in Config.STOP_WORDS]
                if keywords:
                    self.context_summary = f"用户正在讨论{keywords[0]}等话题"
                else:
                    self.context_summary = "用户正在进行对话"
    
    def _extract_key_entities(self, text):
        """从文本中提取关键实体"""
        words = pseg.cut(text)
        entities = []
        
        for word, flag in words:
            if flag.startswith('n') and word not in Config.STOP_WORDS and len(word) > 1:
                entities.append(word)
        
        # 按提及频率排序
        entity_counts = {}
        for entity in entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        return sorted(entity_counts.keys(), key=lambda x: entity_counts[x], reverse=True)
    
    def _update_entity_graph_with_coreference(self, query):
        """使用共指消解更新实体关系图"""
        # 提取当前问题中的实体
        current_entities = self.get_entities_with_types(query)
        
        # 更新实体提及记录
        for entity, entity_type in current_entities:
            self.entity_mentions[entity].append({
                "turn": len(self.history),
                "query": query,
                "type": entity_type
            })
        
        # 如果没有实体但有代词，保持当前焦点
        if not current_entities and self.current_focus:
            # 将当前焦点添加到当前实体中
            current_entities = [(self.current_focus, "common")]
        
        # 将实体添加到关系图中
        for entity, entity_type in current_entities:
            if entity not in self.entity_graph:
                self.entity_graph[entity] = {
                    "mentioned": 1, 
                    "related": set(),
                    "type": entity_type,
                    "last_mentioned": len(self.history)
                }
            else:
                self.entity_graph[entity]["mentioned"] += 1
                self.entity_graph[entity]["last_mentioned"] = len(self.history)
        
        # 建立实体间关系（同一问题中提到的实体互相关联）
        for i in range(len(current_entities)):
            for j in range(i+1, len(current_entities)):
                entity1, type1 = current_entities[i]
                entity2, type2 = current_entities[j]
                self.entity_graph[entity1]["related"].add((entity2, type2))
                self.entity_graph[entity2]["related"].add((entity1, type1))
        
        # 更新当前焦点
        if current_entities:
            self.current_focus = current_entities[0][0]
            self.focus_history.append(current_entities[0][0])
    
    def get_entities_from_query(self, query):
        """从查询中提取实体"""
        entities_with_types = self.get_entities_with_types(query)
        return [entity for entity, entity_type in entities_with_types]
    
    def get_response_from_turn(self, turn_index):
        """获取特定轮次的系统响应"""
        if 0 <= turn_index < len(self.history):
            turn = self.history[turn_index]
            return turn.get("system_response", "")
        return ""
    
    def get_entities_from_turn(self, turn_index):
        """获取特定轮次的实体"""
        if 0 <= turn_index < len(self.history):
            return self.get_entities_from_query(self.get_query_from_turn(turn_index))
        return []
    
    def get_last_entities(self):
        """获取上一轮对话的实体"""
        if not self.history:
            return []
        return self.get_entities_from_query(self.get_last_query())
    
    def get_context_for_retrieval(self):
        """获取检索上下文"""
        if self.context_summary:
            return self.context_summary
        elif self.current_focus:
            return self.current_focus
        elif self.history:
            return self.get_last_query()
        return ""
    
    def add_to_history(self, user_query, system_response, parsed_info):
        """将本轮对话加入历史"""
        self.history.append({
            "user_query": user_query,
            "system_response": system_response,
            "parsed_info": parsed_info
        })
        
        # 更新当前焦点
        entities = self.get_entities_from_query(user_query)
        if entities:
            self.current_focus = entities[0]
            self.focus_history.append(entities[0])
    
    def get_last_query(self):
        """获取上一轮查询"""
        if not self.history:
            return ""
        return self.get_query_from_turn(-1)
    
    def get_query_from_turn(self, index):
        """从历史记录中获取查询内容"""
        turn = self.history[index]
        # 兼容两种键名格式
        if "query" in turn:
            return turn["query"]
        return turn["user_query"]
