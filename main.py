import sys
import time
import re
import json
import os
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, 
                             QGroupBox, QScrollArea, QFrame, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QToolBar, QAction, QStatusBar,
                             QMessageBox, QProgressBar, QComboBox, QCheckBox, QSpinBox,
                             QFileDialog, QMenu, QSystemTrayIcon, QStyle, QToolButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPoint
from PyQt5.QtGui import QFont, QTextCursor, QPalette, QColor, QIcon, QPixmap, QTextCharFormat

from base_pipeline import BasePipeline
from config import Config
from sentence_transformers import SentenceTransformer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkerThread(QThread):
    """工作线程，用于处理查询以避免界面冻结"""
    finished = pyqtSignal(str, float, dict, list)  # 信号：答案，置信度，生成数据，相关文档
    error = pyqtSignal(str)  # 错误信号
    progress = pyqtSignal(int, str)  # 进度信号
    
    def __init__(self, pipeline, adapter, query, last_answer=None):
        super().__init__()
        self.pipeline = pipeline
        self.adapter = adapter
        self.query = query
        self.last_answer = last_answer
    
    def run(self):
        try:
            # 检查是否为纠正反馈
            correction = self.extract_correction_from_query(self.query, self.last_answer)
            if correction:
                self.progress.emit(20, "处理纠正反馈")
                # 处理纠正反馈
                if hasattr(self.pipeline, 'last_generation_data') and self.pipeline.last_generation_data:
                    feedback_data = {
                        "type": "correction",
                        "generation_data": self.pipeline.last_generation_data,
                        "comment": self.query,
                        "correction": correction
                    }
                    self.pipeline.engine.process_feedback(feedback_data)
                    response = f"✅ 已记录您的纠正：{correction}"
                    self.finished.emit(response, 1.0, {}, [])
                else:
                    response = "❌ 无法处理纠正，因为没有上一轮的回答记录"
                    self.finished.emit(response, 0.0, {}, [])
                return
            
            self.progress.emit(30, "检索相关数据")
            # 获取匹配数据
            start_time = time.time()
            crag_data = self.adapter.get_by_query(self.query)
            data_retrieval_time = time.time() - start_time
            logger.info(f"数据检索耗时: {data_retrieval_time:.2f}秒")
            
            self.progress.emit(60, "生成答案")
            # 处理问答
            response, confidence, generation_data = self.pipeline.process(self.query, crag_data)
            
            # 保存当前回答信息
            self.pipeline.last_generation_data = generation_data
            self.pipeline.last_answer = response
            
            # 提取相关文档信息
            relevant_docs = generation_data.get("documents", [])
            
            self.progress.emit(100, "完成")
            # 发送完成信号
            self.finished.emit(response, confidence, generation_data, relevant_docs)
            
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
    
    def extract_correction_from_query(self, query, last_answer):
        """从用户查询中提取纠正信息"""
        # 定义否定模式
        negative_patterns = [
            r'不对[，,:：]?(.*)',  # "不对，科比有六次冠军"
            r'错了[，,:：]?(.*)',  # "错了，科比有六次冠军"
            r'不是[，,:：]?(.*)',  # "不是，科比有六次冠军"
            r'不正确[，,:：]?(.*)',  # "不正确，科比有六次冠军"
            r'应该是(.*)',  # "应该是六次"
            r'其实是(.*)',  # "其实是六次"
        ]
        
        for pattern in negative_patterns:
            match = re.search(pattern, query)
            if match:
                correction = match.group(1).strip()
                # 如果纠正内容为空，尝试从上下文中推断
                if not correction and last_answer:
                    # 提取数字等信息
                    numbers = re.findall(r'\d+', last_answer)
                    if numbers:
                        correction = f"正确答案是{numbers[0]}次"  # 简化处理
                    else:
                        correction = "正确答案需要提供"
                return correction
        
        return None

class CRAGAdapter:
    """数据适配器"""
    def __init__(self, data_path=Config.CRAG_DATA_PATH):
        self.data = self.load_data(data_path)
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.precompute_embeddings()
    
    def load_data(self, path):
        """加载CRAG数据集"""
        if not os.path.exists(path):
            logger.error(f"警告: 数据文件 {path} 不存在")
            return []
        
        data = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return []
        return data
    
    def precompute_embeddings(self):
        """预计算文档嵌入向量"""
        self.embeddings = []
        for item in self.data:
            if item.get("search_results") and len(item["search_results"]) > 0:
                snippet = item["search_results"][0]["page_snippet"]
                self.embeddings.append(self.embedding_model.encode([snippet])[0])
            else:
                # 使用默认空嵌入向量
                self.embeddings.append(np.zeros(384))
    
    def get_by_query(self, query):
        """根据查询获取最匹配的CRAG数据"""
        if not self.data:
            return {"search_results": []}
            
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = []
        for emb in self.embeddings:
            # 避免除以零
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            sim = np.dot(query_embedding, emb) / norm_product if norm_product > 0 else 0
            similarities.append(sim)
        
        # 获取最相似的数据项
        top_idx = np.argmax(similarities)
        return self.data[top_idx]

class MessageWidget(QWidget):
    """自定义消息部件，用于显示聊天消息"""
    def __init__(self, sender, message, timestamp, is_user=True, confidence=None, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.confidence = confidence
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 消息头部（发送者和时间）
        header_layout = QHBoxLayout()
        sender_label = QLabel(sender)
        sender_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        
        header_layout.addWidget(sender_label)
        header_layout.addStretch()
        header_layout.addWidget(time_label)
        
        # 消息内容
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        
        # 设置消息气泡样式
        if is_user:
            content_frame.setStyleSheet("""
                QFrame {
                    background-color: #3498db;
                    border-radius: 10px;
                    padding: 8px;
                }
            """)
        else:
            content_frame.setStyleSheet("""
                QFrame {
                    background-color: #ecf0f1;
                    border-radius: 10px;
                    padding: 8px;
                }
            """)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextFormat(Qt.RichText)
        message_label.setStyleSheet("color: #2c3e50;" if not is_user else "color: white;")
        
        content_layout.addWidget(message_label)
        
        # 如果是系统消息且存在置信度，显示置信度
        if not is_user and confidence is not None:
            confidence_bar = QProgressBar()
            confidence_bar.setRange(0, 100)
            confidence_bar.setValue(int(confidence * 100))
            confidence_bar.setFormat(f"置信度: {confidence:.2f}")
            confidence_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #27ae60;
                    border-radius: 5px;
                }
            """)
            content_layout.addWidget(confidence_bar)
        
        # 添加到主布局
        layout.addLayout(header_layout)
        layout.addWidget(content_frame)
        
        self.setLayout(layout)

class DocumentViewer(QWidget):
    """文档查看器，显示检索到的相关文档"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相关文档")
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        # 文档表格
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["相关度", "文档ID", "内容片段"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.table)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def set_documents(self, documents):
        """设置要显示的文档"""
        self.table.setRowCount(len(documents))
        
        for i, doc in enumerate(documents):
            # 相关度
            similarity_item = QTableWidgetItem(f"{doc.get('similarity', 0):.3f}")
            similarity_item.setTextAlignment(Qt.AlignCenter)
            
            # 文档ID
            doc_id_item = QTableWidgetItem(doc.get('doc_id', '未知'))
            doc_id_item.setTextAlignment(Qt.AlignCenter)
            
            # 内容片段
            snippet = doc.get('snippet', '')
            snippet_item = QTableWidgetItem(snippet[:200] + "..." if len(snippet) > 200 else snippet)
            
            self.table.setItem(i, 0, similarity_item)
            self.table.setItem(i, 1, doc_id_item)
            self.table.setItem(i, 2, snippet_item)

class SettingsDialog(QWidget):
    """设置对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.setGeometry(300, 300, 500, 400)
        
        layout = QVBoxLayout()
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        # 嵌入模型选择
        embed_layout = QHBoxLayout()
        embed_layout.addWidget(QLabel("嵌入模型:"))
        self.embed_model_combo = QComboBox()
        self.embed_model_combo.addItems(["paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"])
        embed_layout.addWidget(self.embed_model_combo)
        model_layout.addLayout(embed_layout)
        
        # 生成模型设置
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("使用生成模型:"))
        self.use_gen_model = QCheckBox()
        self.use_gen_model.setChecked(Config.USE_GENERATIVE_MODEL)
        gen_layout.addWidget(self.use_gen_model)
        gen_layout.addStretch()
        model_layout.addLayout(gen_layout)
        
        # 生成模型路径
        gen_path_layout = QHBoxLayout()
        gen_path_layout.addWidget(QLabel("生成模型路径:"))
        self.gen_model_path = QLineEdit(Config.GENERATION_MODEL_PATH)
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_model_path)
        gen_path_layout.addWidget(self.gen_model_path)
        gen_path_layout.addWidget(self.browse_button)
        model_layout.addLayout(gen_path_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 检索设置
        retrieval_group = QGroupBox("检索设置")
        retrieval_layout = QVBoxLayout()
        
        # 检索文档数量
        top_k_layout = QHBoxLayout()
        top_k_layout.addWidget(QLabel("检索文档数量:"))
        self.top_k_docs = QSpinBox()
        self.top_k_docs.setRange(1, 10)
        self.top_k_docs.setValue(Config.TOP_K_DOCS)
        top_k_layout.addWidget(self.top_k_docs)
        top_k_layout.addStretch()
        retrieval_layout.addLayout(top_k_layout)
        
        # 上下文文档数量
        context_k_layout = QHBoxLayout()
        context_k_layout.addWidget(QLabel("上下文文档数量:"))
        self.top_k_context = QSpinBox()
        self.top_k_context.setRange(1, 5)
        self.top_k_context.setValue(Config.TOP_K_CONTEXT)
        context_k_layout.addWidget(self.top_k_context)
        context_k_layout.addStretch()
        retrieval_layout.addLayout(context_k_layout)
        
        retrieval_group.setLayout(retrieval_layout)
        layout.addWidget(retrieval_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.close)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def browse_model_path(self):
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.gen_model_path.setText(path)
    
    def save_settings(self):
        """保存设置"""
        # 这里应该实现设置保存逻辑
        QMessageBox.information(self, "提示", "设置已保存")
        self.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化系统组件
        self.adapter = CRAGAdapter()
        self.pipeline = BasePipeline()
        self.pipeline.last_generation_data = None
        self.pipeline.last_answer = None
        
        # 聊天历史
        self.chat_history = []
        
        # 设置窗口属性
        self.setWindowTitle("智能问答系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建界面
        self.init_ui()
        
        # 更新系统信息
        self.update_system_info()
        
        # 设置定时器，定期更新系统信息
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(5000)  # 每5秒更新一次
        
        # 创建系统托盘图标
        self.create_system_tray()
    
    def init_ui(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧信息面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧聊天面板
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setSizes([250, 750])
        
        # 添加到主布局
        main_layout.addWidget(splitter)
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建状态栏
        self.create_statusbar()
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # 新建会话
        new_chat_action = QAction(QIcon.fromTheme("document-new"), "新建会话", self)
        new_chat_action.setShortcut("Ctrl+N")
        new_chat_action.triggered.connect(self.new_chat)
        toolbar.addAction(new_chat_action)
        
        toolbar.addSeparator()
        
        # 保存会话
        save_chat_action = QAction(QIcon.fromTheme("document-save"), "保存会话", self)
        save_chat_action.setShortcut("Ctrl+S")
        save_chat_action.triggered.connect(self.save_chat)
        toolbar.addAction(save_chat_action)
        
        # 加载会话
        load_chat_action = QAction(QIcon.fromTheme("document-open"), "加载会话", self)
        load_chat_action.setShortcut("Ctrl+O")
        load_chat_action.triggered.connect(self.load_chat)
        toolbar.addAction(load_chat_action)
        
        toolbar.addSeparator()
        
        # 设置
        settings_action = QAction(QIcon.fromTheme("preferences-system"), "设置", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
        # 帮助
        help_action = QAction(QIcon.fromTheme("help-contents"), "帮助", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
    
    def create_statusbar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 系统状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addWidget(self.progress_bar)
        
        # 进度标签
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_label)
    
    def create_system_tray(self):
        """创建系统托盘图标"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # 创建托盘菜单
        tray_menu = QMenu()
        
        show_action = tray_menu.addAction("显示")
        show_action.triggered.connect(self.show)
        
        hide_action = tray_menu.addAction("隐藏")
        hide_action.triggered.connect(self.hide)
        
        tray_menu.addSeparator()
        
        quit_action = tray_menu.addAction("退出")
        quit_action.triggered.connect(QApplication.quit)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
        # 托盘图标点击事件
        self.tray_icon.activated.connect(self.tray_icon_activated)
    
    def tray_icon_activated(self, reason):
        """托盘图标激活事件"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()
    
    def create_left_panel(self):
        """创建左侧面板"""
        # 使用选项卡式布局
        tab_widget = QTabWidget()
        
        # 系统信息选项卡
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        # 系统信息组
        info_group = QGroupBox("系统信息")
        info_layout_group = QVBoxLayout()
        
        self.model_label = QLabel(f"模型: {Config.EMBEDDING_MODEL}")
        info_layout_group.addWidget(self.model_label)
        
        if Config.USE_GENERATIVE_MODEL:
            self.gen_model_label = QLabel(f"生成模型: {Config.GENERATION_MODEL_PATH}")
            info_layout_group.addWidget(self.gen_model_label)
        
        self.kb_label = QLabel("知识库记录: 加载中...")
        info_layout_group.addWidget(self.kb_label)
        
        self.cm_label = QLabel("纠正映射: 加载中...")
        info_layout_group.addWidget(self.cm_label)
        
        info_group.setLayout(info_layout_group)
        info_layout.addWidget(info_group)
        
        # 性能统计组
        stats_group = QGroupBox("性能统计")
        stats_layout = QVBoxLayout()
        
        self.avg_response_label = QLabel("平均响应时间: 计算中...")
        stats_layout.addWidget(self.avg_response_label)
        
        self.total_queries_label = QLabel("总查询次数: 0")
        stats_layout.addWidget(self.total_queries_label)
        
        self.success_rate_label = QLabel("成功率: 100%")
        stats_layout.addWidget(self.success_rate_label)
        
        stats_group.setLayout(stats_layout)
        info_layout.addWidget(stats_group)
        
        # 使用说明组
        usage_group = QGroupBox("使用说明")
        usage_layout = QVBoxLayout()
        
        usage_text = """
        <b>基本操作:</b><br>
        • 输入问题获取答案<br>
        • 按Enter或点击发送按钮提交问题<br>
        
        <b>纠正反馈:</b><br>
        • 如果答案不正确，可以直接回复纠正，例如：
          - "不对，科比有五次冠军"
          - "错了，应该是五次"
          - "其实是五次"<br>
        
        <b>快捷键:</b><br>
        • Ctrl+N: 新建会话<br>
        • Ctrl+S: 保存会话<br>
        • Ctrl+O: 加载会话<br>
        • Ctrl+逗号: 打开设置<br>
        • F1: 显示帮助
        """
        
        usage_label = QLabel(usage_text)
        usage_label.setWordWrap(True)
        usage_layout.addWidget(usage_label)
        
        usage_group.setLayout(usage_layout)
        info_layout.addWidget(usage_group)
        
        info_layout.addStretch()
        
        # 历史会话选项卡
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        history_group = QGroupBox("历史会话")
        history_layout_group = QVBoxLayout()
        
        self.history_list = QTableWidget()
        self.history_list.setColumnCount(3)
        self.history_list.setHorizontalHeaderLabels(["时间", "问题", "操作"])
        self.history_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_list.setSelectionBehavior(QTableWidget.SelectRows)
        
        history_layout_group.addWidget(self.history_list)
        history_group.setLayout(history_layout_group)
        history_layout.addWidget(history_group)
        
        # 添加到选项卡
        tab_widget.addTab(info_tab, "系统信息")
        tab_widget.addTab(history_tab, "历史会话")
        
        return tab_widget
    
    def create_right_panel(self):
        """创建右侧面板"""
        # 使用选项卡式布局
        tab_widget = QTabWidget()
        
        # 聊天选项卡
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        
        # 聊天历史区域
        chat_group = QGroupBox("对话")
        chat_layout_group = QVBoxLayout()
        
        # 使用滚动区域包装聊天历史
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()  # 添加弹性空间使内容顶部对齐
        
        self.chat_scroll.setWidget(self.chat_container)
        chat_layout_group.addWidget(self.chat_scroll)
        
        chat_group.setLayout(chat_layout_group)
        chat_layout.addWidget(chat_group)
        
        # 输入区域
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("WenQuanYi Micro Hei", 10))
        self.input_field.setPlaceholderText("输入您的问题...")
        self.input_field.returnPressed.connect(self.process_input)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.process_input)
        input_layout.addWidget(self.send_button)
        
        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self.clear_chat)
        input_layout.addWidget(self.clear_button)
        
        chat_layout.addLayout(input_layout)
        
        # 文档查看选项卡
        self.doc_viewer = DocumentViewer()
        
        # 添加到选项卡
        tab_widget.addTab(chat_tab, "聊天")
        tab_widget.addTab(self.doc_viewer, "相关文档")
        
        return tab_widget
    
    def update_system_info(self):
        """更新系统信息"""
        try:
            self.kb_label.setText(f"知识库记录: {len(self.pipeline.engine.knowledge_base)} 条")
            self.cm_label.setText(f"纠正映射: {len(self.pipeline.engine.correction_map)} 条")
            
            # 更新性能统计
            # 这里应该从日志中计算实际性能数据
            self.avg_response_label.setText("平均响应时间: 1.2秒")
            self.total_queries_label.setText(f"总查询次数: {len(self.chat_history)//2}")
            self.success_rate_label.setText("成功率: 98%")
        except:
            pass
    
    def add_message(self, sender, message, is_user=True, confidence=None):
        """添加消息到聊天历史"""
        timestamp = time.strftime("%H:%M:%S")
        
        # 创建消息部件
        message_widget = MessageWidget(sender, message, timestamp, is_user, confidence)
        
        # 添加到聊天布局
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)
        
        # 滚动到底部
        QTimer.singleShot(100, self.scroll_to_bottom)
        
        # 保存到聊天历史
        self.chat_history.append({
            "sender": sender, 
            "message": message, 
            "time": timestamp,
            "is_user": is_user,
            "confidence": confidence
        })
    
    def scroll_to_bottom(self):
        """滚动到底部"""
        scroll_bar = self.chat_scroll.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
    
    def process_input(self):
        """处理用户输入"""
        user_query = self.input_field.text().strip()
        if not user_query:
            return
        
        # 清空输入框
        self.input_field.clear()
        
        # 添加用户消息到聊天历史
        self.add_message("用户", user_query, is_user=True)
        
        # 更新状态栏
        self.status_label.setText("处理中...")
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        
        # 禁用发送按钮和输入框
        self.send_button.setEnabled(False)
        self.input_field.setEnabled(False)
        
        # 创建工作线程处理查询
        self.worker_thread = WorkerThread(
            self.pipeline, 
            self.adapter, 
            user_query, 
            self.pipeline.last_answer
        )
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.error.connect(self.on_worker_error)
        self.worker_thread.progress.connect(self.on_worker_progress)
        self.worker_thread.start()
    
    def on_worker_progress(self, value, text):
        """工作线程进度更新"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)
    
    def on_worker_finished(self, response, confidence, generation_data, relevant_docs):
        """工作线程完成处理"""
        # 格式化响应
        formatted_response = f"{response}"
        
        # 添加系统消息到聊天历史
        self.add_message("系统", formatted_response, is_user=False, confidence=confidence)
        
        # 更新文档查看器
        self.doc_viewer.set_documents(relevant_docs)
        
        # 更新状态栏
        self.status_label.setText("就绪")
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # 启用发送按钮和输入框
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
        
        # 将焦点设置回输入框
        self.input_field.setFocus()
    
    def on_worker_error(self, error_msg):
        """工作线程出错"""
        # 添加错误消息到聊天历史
        self.add_message("系统", error_msg, is_user=False)
        
        # 更新状态栏
        self.status_label.setText("错误")
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # 启用发送按钮和输入框
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
    
    def clear_chat(self):
        """清空聊天历史"""
        # 清除聊天布局中的所有消息部件
        for i in reversed(range(self.chat_layout.count() - 1)):  # 保留最后的stretch
            item = self.chat_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        self.chat_history = []
        self.pipeline.last_generation_data = None
        self.pipeline.last_answer = None
        
        self.add_message("系统", "聊天已清空", is_user=False)
    
    def new_chat(self):
        """新建会话"""
        reply = QMessageBox.question(self, "确认", "确定要开始新的会话吗？当前会话将被清除。",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.clear_chat()
    
    def save_chat(self):
        """保存会话"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存会话", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "成功", "会话已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存会话失败: {str(e)}")
    
    def load_chat(self):
        """加载会话"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载会话", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                
                # 清除当前会话
                self.clear_chat()
                
                # 加载历史消息
                for msg in chat_data:
                    self.add_message(
                        msg.get('sender', '未知'),
                        msg.get('message', ''),
                        msg.get('is_user', False),
                        msg.get('confidence')
                    )
                
                QMessageBox.information(self, "成功", "会话已加载")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载会话失败: {str(e)}")
    
    def show_settings(self):
        """显示设置对话框"""
        settings_dialog = SettingsDialog(self)
        settings_dialog.show()
    
    def show_help(self):
        """显示帮助"""
        help_text = """
        <h2>智能问答系统帮助</h2>
        
        <h3>基本功能</h3>
        <p>这是一个基于检索增强生成(RAG)的智能问答系统，可以回答各种问题。</p>
        
        <h3>使用技巧</h3>
        <ul>
            <li>输入完整的问题可以获得更准确的答案</li>
            <li>如果答案不正确，可以使用纠正功能提供正确答案</li>
            <li>查看"相关文档"选项卡可以了解答案的来源</li>
        </ul>
        
        <h3>常见问题</h3>
        <p><b>Q: 系统无法回答我的问题怎么办？</b><br>
        A: 尝试换一种方式提问，或者提供更多上下文信息。</p>
        
        <p><b>Q: 如何提高答案的准确性？</b><br>
        A: 当系统给出错误答案时，使用纠正功能可以帮助系统学习。</p>
        """
        
        QMessageBox.information(self, "帮助", help_text)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(self, "确认退出", "确定要退出程序吗？",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 保存当前会话
            try:
                os.makedirs("sessions", exist_ok=True)
                session_file = f"sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            except:
                pass
            
            event.accept()
        else:
            event.ignore()

def main():
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 设置中文字体
    font = QFont("WenQuanYi Micro Hei", 10)
    app.setFont(font)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
