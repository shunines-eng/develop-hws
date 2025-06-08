# main.py
from itertools import chain
import sys
import threading
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QSplitter,
                             QTreeView, QPlainTextEdit, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QLabel, QFileSystemModel,
                             QFileDialog, QScrollArea, QSizePolicy, QStatusBar,
                             QDialog
                             )
from PyQt5.QtCore import Qt, QDir, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import requests
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QComboBox, QTextEdit  # 新增导入
from PyQt5.Qsci import QsciLexerJava  # Java语法支持
from PyQt5.QtCore import QThread
# ==================== API 客户端 ====================
class APIClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.current_project = None

    # 调用后端打开文件
    def open_project(self, project_path):
        try:
            resp_open = self._post("/openFilePath", {"file_path": str(project_path)})
            if not isinstance(resp_open, dict) or resp_open.get("status") != "success":
                raise ValueError("Invalid open response")

            resp_doc = self._get("/askDemo")
            if not isinstance(resp_doc, dict):
                raise ValueError("askDemo返回非字典数据")

            return {
                "open_status": resp_open,
                "doc_status": resp_doc
            }
        except Exception as e:
            return {"status": "error", "message": f"前端错误: {str(e)}"}

    # 更新知识库
    def save_and_update(self):
        """触发知识库更新"""
        return self._get("/update_dataBase")

    def ask_question(self, question):
        """统一请求实现"""
        try:
            resp = requests.post(
                f"{self.base_url}/ask",
                json={"question": question},
                timeout=210  # 统一超时时间
            )
            resp.raise_for_status()

            # 强化响应检查
            result = resp.json()
            if not isinstance(result, dict):
                raise ValueError("返回非字典数据")

            if "status" not in result:
                raise KeyError("响应缺少status字段")

            return result

        except Exception as e:
            print(f"[ERROR] 提问请求异常: {str(e)}")
            return {
                "status": "error",
                "message": f"通信失败: {type(e).__name__}"
            }

        # return self._post(
        #     f"{self.base_url}/ask",
        #     json={"question": question},
        #     timeout=240
        # )

    def generate_commit_message(self):
        """获取生成的提交信息"""
        try:
            resp = requests.get(f"{self.base_url}/commit_create", timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def perform_git_commit(self, message):
        """执行Git提交"""
        try:
            resp = requests.post(
                f"{self.base_url}/git_commit",
                json={"message": message},
                timeout=210
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e1:
            return {"status": "error", "message": str(e1)}

    def clone_repository(self, repo_url, save_path):
        """克隆Git仓库"""
        try:
            resp = requests.post(
                f"{self.base_url}/clone_repo",
                json={"repo_url": repo_url, "save_path": save_path},
                timeout=300  # 克隆可能需要较长时间
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _post(self, url, data):
        response = requests.post(self.base_url + url, json=data)
        response.raise_for_status()  # 非 2xx 状态码会抛出 HTTPError
        return response.json()

    def _get(self, url):
        response = requests.get(self.base_url + url)
        response.raise_for_status()
        return response.json()

# 编译器组件库
from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciLexerHTML, QsciLexerMarkdown

# 基于QScintilla的代码编辑器组件
class CodeEditor(QsciScintilla):
    def __init__(self, language="python"):
        super().__init__()
        self.setFont(QFont("Consolas", 12))
        self.setMarginWidth(0, "000")
        self.setMarginType(0, QsciScintilla.NumberMargin)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)
        self.setAutoIndent(True)
        self.setStyleSheet("background: white;")  # 只设置背景，文本颜色由lexer控制
        self.setCaretLineBackgroundColor(QColor("#2d2d30"))
        self.set_lexer(language)

    def set_lexer(self, lang):
        lexer = {
            "python": QsciLexerPython(),
            "html": QsciLexerHTML(),
            "markdown": QsciLexerMarkdown(),
            "java": QsciLexerJava()
        }.get(lang, QsciLexerPython())

        lexer.setDefaultFont(QFont("Consolas", 12))
        lexer.setDefaultColor(QColor("#000000"))   # 改为黑色
        self.setLexer(lexer)

from PyQt5.QtWidgets import QTextBrowser  # 确保已导入

# class A(B) 相当于设置一个A类，作为B的子类
class ChatMessage(QWidget):
    # 聊天信息气泡组件
    def __init__(self, is_ai=False, markdown_text=""):
        # 显示调用父类函数，使得布局、渲染等父类方法生效
        super().__init__()
        self.content = QTextBrowser()
        # QWidget通用设置
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # 在初始化时调用样式设置方法
        self._apply_bubble_style(is_ai)  # 新增此行

        # 主布局
        # 布局配置
        layout = QHBoxLayout(self)
        # 内容高度计算
        layout.setContentsMargins(30, 8, 30, 8)
        # 组件排列
        layout.setSpacing(8)

        # 文本浏览器，使用QTBrowser显示富文本内容
        self.content = QTextBrowser()
        self.content.setMarkdown(markdown_text)
        self.content.setOpenExternalLinks(True)  # 允许打开外部链接

        # 必须在调用前定义方法
        self._calculate_content_height()

        self._arrange_components(is_ai)

    def _apply_bubble_style(self, is_ai):
        # 强制清除可能继承的样式
        self.content.setStyleSheet("")

        # 修改边框样式
        bubble_css = f"""
            QTextBrowser {{
                background: {'#E8F4FF' if is_ai else '#F0F8E9'} !important;
                color: #333333;
                border-radius: 15px;
                padding: 12px 4px;
                border: 2px solid {'#2A7FB4' if is_ai else '#388E3C'} !important;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                line-height: 1.6;
                margin: 8px 0;
                qproperty-alignment: AlignLeft;
            }}
            QTextBrowser:hover {{
                border: 2px solid {'#1A6FB4' if is_ai else '#288E3C'};
            }}
        """
        self.content.setStyleSheet(bubble_css)
        self.content.setAutoFillBackground(True)  # 关键：启用背景填充
        self.content.ensurePolished()  # 立即应用样式

    # 确保方法正确定义在类内部
    def _calculate_content_height(self):
        # 确保组件已完成初始布局
        self.content.document().documentLayout().documentSizeChanged.connect(
            self._update_height_after_layout
        )

    def _update_height_after_layout(self):
        """在布局完成后触发的精确高度计算"""
        doc = self.content.document()
        avail_width = self.content.width() - 6  # 更通用的边距补偿
        doc.setTextWidth(avail_width)

        # 获取实际内容高度
        content_height = doc.documentLayout().documentSize().height()
        ideal_height = int(content_height) + 24  # 增加padding

        # 设置高度限制（保留最小高度避免折叠）
        self.content.setMinimumHeight(min(ideal_height, 12000))
        self.content.setMaximumHeight(12000)

    def _arrange_components(self, is_ai):
        layout = self.layout()
        # 清理旧组件（保留content）
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() and item.widget() != self.content:
                item.widget().deleteLater()

        # 创建头像并强制顶部对齐
        avatar = self._avatar("AI" if is_ai else "我", "#4EC9B0" if is_ai else "#C586C0")
        avatar.setAlignment(Qt.AlignTop)  # 关键修改：设置顶部对齐

        # 动态排列组件
        if is_ai:
            components = [avatar, self.content, QWidget()]
        else:
            components = [QWidget(), self.content, avatar]

        # 添加组件到布局
        for widget in components:
            if widget != self.content:
                widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            layout.addWidget(widget, alignment=Qt.AlignTop)  # 每个组件都顶部对齐

    def _avatar(self, name, color):
        """创建带圆形背景的头像标签"""
        avatar = QLabel(name[0].upper())  # 显示首字母
        avatar.setFixedSize(36, 36)  # 固定尺寸

        # 圆形样式
        avatar.setStyleSheet(f"""
            QLabel {{
                background: {color};
                color: white;
                border-radius: 18px;
                font-family: 'Microsoft YaHei';
                font-size: 16px;
                font-weight: bold;
                qproperty-alignment: AlignCenter;
            }}
        """)
        return avatar

# ==================== 主窗口 ====================
class IDEWindow(QMainWindow):
    update_status = pyqtSignal(str, str) # 状态栏更新信号（消息，颜色）
    add_chat_message = pyqtSignal(str, bool) # 添加聊天信息
    update_file_tree = pyqtSignal(str) # 更新文件树
    clear_chat_display = pyqtSignal() # 清空聊天区
    load_history_complete = pyqtSignal() # 历史记录加载完成
    ask_response = pyqtSignal(dict)  # AI响应结果（线程安全）

    def __init__(self):
        # 关键组件
        super().__init__() #？
        self._bottom_spacer = None
        self.send_btn = None # 发送按钮
        self.current_file_label = None # 当前文件路径显示
        self.save_btn = None
        self.project_path = None # 项目路径输入框
        self.loading_msg = None
        self.client = APIClient()
        self.current_project = None
        self.current_file = None
        self.model = None
        self.init_ui()
        self.init_signals()
        self.setWindowTitle("AI Coding Assistant")
        self.resize(1600, 900)
        self.load_history_complete.connect(self.on_history_loaded)
        self.clear_chat_display.connect(self._clear_chat)
        self.ask_response.connect(self._handle_ask_response)
        # 添加生成状态标志(用于进行对AI提问时对用户的反馈)
        self.is_generating = False

    def on_history_loaded(self):
        # 增加类型检查和更安全的删除逻辑
        loading_msg = getattr(self, 'loading_msg', None)
        if isinstance(loading_msg, QWidget):
            try:
                loading_msg.setParent(None)
                loading_msg.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.loading_msg = None

    def _clear_chat(self):
        """安全清空聊天区域"""
        for i in reversed(range(self.chat_layout.count())):
            widget = self.chat_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 顶部项目栏
        main_layout.addWidget(self.create_top_bar())

        # 主内容区
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.create_file_tree())
        main_splitter.addWidget(self.create_editor_panel())
        main_splitter.addWidget(self.create_chat_panel())
        main_splitter.setSizes([300, 800, 500])

        main_layout.addWidget(main_splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setStatusBar(QStatusBar())

    def init_signals(self):
        self.update_status.connect(self.handle_status_update)
        self.add_chat_message.connect(self.handle_new_message)
        self.update_file_tree.connect(self.handle_file_tree_update)
        
    def create_top_bar(self):
        bar = QWidget()
        bar.setMaximumHeight(60)
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        self.project_path = QLineEdit()
        self.project_path.setPlaceholderText("选择或输入项目路径...")
        self.project_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_style = "QPushButton { min-width: 80px; }"
        browse_btn = QPushButton("浏览")
        load_btn = QPushButton("加载项目")
        browse_btn.setStyleSheet(btn_style)
        load_btn.setStyleSheet(btn_style)

        # 添加克隆仓库按钮
        clone_btn = QPushButton("克隆仓库")
        clone_btn.setStyleSheet("background: #2196F3; color: white;")
        clone_btn.setToolTip("从Git仓库克隆项目")

        browse_btn.setStyleSheet(btn_style)
        load_btn.setStyleSheet(btn_style)
        clone_btn.setStyleSheet(btn_style + "background: #2196F3; color: white;")

        browse_btn.clicked.connect(self.browse_project)
        load_btn.clicked.connect(self.load_project)
        clone_btn.clicked.connect(self.clone_repository)  # 连接克隆方法

        # 将所有上述实现的按钮渲染到布局中
        layout.addWidget(QLabel("项目路径:"))
        layout.addWidget(self.project_path)
        layout.addWidget(browse_btn)
        layout.addWidget(load_btn)
        layout.addWidget(clone_btn)
        bar.setLayout(layout)
        return bar

    def create_file_tree(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # 文件树
        self.file_tree = QTreeView() # 文件目录树
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setAnimated(True)
        self.file_tree.doubleClicked.connect(self.open_file)

        # 提交信息板块
        commit_group = QWidget()
        commit_layout = QVBoxLayout()
        commit_layout.setContentsMargins(8, 8, 8, 8)

        self.gen_commit_btn = QPushButton("生成提交信息")
        self.gen_commit_btn.clicked.connect(self.generate_commit_msg)

        self.commit_edit = QPlainTextEdit()
        self.commit_edit.setPlaceholderText("提交信息将在此显示...")
        self.commit_edit.setStyleSheet("""
            QPlainTextEdit { 
                background: white; 
                color: #333333; 
                border: 1px solid #ccc;
                min-height: 80px;
            }""")

        self.do_commit_btn = QPushButton("执行提交")
        self.do_commit_btn.clicked.connect(self.execute_commit)

        commit_layout.addWidget(QLabel("提交操作:"))
        commit_layout.addWidget(self.gen_commit_btn)
        commit_layout.addWidget(self.commit_edit)
        commit_layout.addWidget(self.do_commit_btn)
        commit_group.setLayout(commit_layout)

        layout.addWidget(self.file_tree)
        layout.addWidget(commit_group)
        panel.setLayout(layout)
        return panel

    # 整编辑器初始化
    def create_editor_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        # 添加语言切换工具栏
        toolbar = QHBoxLayout()
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Python", "HTML", "Markdown", "Java"])
        self.lang_combo.currentTextChanged.connect(self.change_language)
        toolbar.addWidget(QLabel("语言:"))
        toolbar.addWidget(self.lang_combo)

        # 添加保存按钮（核心修复）
        self.save_btn = QPushButton("保存文件")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #388e3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: #2e7d32;
            }
            QPushButton:disabled {
                background: #616161;
            }
        """)
        self.save_btn.clicked.connect(self.save_file)
        toolbar.addWidget(self.save_btn)

        toolbar.addStretch()  # 右对齐按钮

        layout.addLayout(toolbar)

        # 文件路径标签（关键修复）
        self.current_file_label = QLabel("当前文件：未打开")
        self.current_file_label.setStyleSheet("""
            QLabel {
                color: #569cd6;
                font-size: 12px;
                padding: 4px;
                border-bottom: 1px solid #3c3c3c;
            }
        """)
        layout.addWidget(self.current_file_label)

        # 编辑器
        self.editor = CodeEditor()
        layout.addWidget(self.editor)

        panel.setLayout(layout)
        return panel

    #切换编辑器语言
    def change_language(self, lang):
        lang_map = {
            "Python": "python",
            "HTML": "html",
            "Markdown": "markdown",
            "Java": "java"
        }
        self.editor.set_lexer(lang_map.get(lang, "python"))

    def create_chat_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background: white;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 聊天标题
        title = QLabel("AI 编程助手")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #569cd6;
                padding: 8px 0;
                border-bottom: 2px solid #3c3c3c;
            }
        """)
        layout.addWidget(title)

        # 聊天内容区域
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: white;
            }
            QScrollBar:vertical {
                background: #252526;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #3c3c3c;
                min-height: 20px;
            }
        """)

        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(12)

        # === 关键修改 ===
        # 初始化底部弹性空间（替代 addStretch）
        if not hasattr(self, '_bottom_spacer'):
            self._bottom_spacer = QSpacerItem(
                20, 40,
                QSizePolicy.Minimum,
                QSizePolicy.Expanding
            )
            self.chat_layout.addItem(self._bottom_spacer)

        self.chat_scroll.setWidget(self.chat_content)
        layout.addWidget(self.chat_scroll, 1)

        # 输入区域（保持不变）
        input_area = QWidget()
        input_area.setStyleSheet("""
            background: #f8f9fa; 
            border-radius: 6px;
            border: 1px solid #ddd;
            margin-top: 8px;
        """)
        input_layout = QHBoxLayout(input_area)
        input_layout.setContentsMargins(8, 8, 8, 8)

        self.chat_input = QLineEdit()
        self.chat_input.setFixedHeight(48)
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background: #f8f9fa;
                color: #000000;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 14px;
                min-height: 24px;
                qproperty-alignment: AlignVCenter;
            }
        """)
        self.chat_input.returnPressed.connect(self.handle_chat)

        self.send_btn = QPushButton("发送")
        self.send_btn.setToolTip("点击发送问题给AI助手")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
                transition: transform 0.2s;
            }
            QPushButton:hover {
                background: #0062a3;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                transform: scale(0.95);
            }
        """)
        self.send_btn.clicked.connect(self._safe_handle_chat)
        input_layout.addWidget(self.chat_input, 1)
        input_layout.addWidget(self.send_btn)

        layout.addWidget(input_area)

        return panel
    
    def _safe_handle_chat(self):
        try:
            if self.is_generating:  # 防止重复发送
                self.update_status.emit("正在生成回复，请稍候...", "#FF9800")
                return

            question = self.chat_input.text().strip()  # 主线程访问
            if not question:
                self.update_status.emit("请输入问题", "#FF9800")  # 信号安全
                return

            # 用户消息
            self._add_message_safely(f"用户提问：{question}", False)
            # "正在生成回复"提示
            self.waiting_widget = ChatMessage(is_ai=True, markdown_text="▷ AI正在思考...")
            self.chat_layout.addWidget(self.waiting_widget)
            self._scroll_to_bottom()

            self.chat_input.clear()

            # 启动子线程
            threading.Thread(
                target=self._async_send_question,  # 传递函数引用
                args=(question,)  # 传递参数
            ).start()

        except Exception as e:
            # 异常捕获不完整
            print(f"Error: {str(e)}")  # 无日志记录

    def _async_send_question(self, question):
        """执行实际的网络请求"""
        try:
            response = self.client.ask_question(question)
            self.ask_response.emit(response)  # 发射信号
        except requests.exceptions.RequestException as e:
            error = {
                "status": "error",
                "message": f"网络连接失败：{str(e)}"
            }
            self.ask_response.emit(error)
        except Exception as e:
            error = {
                "status": "error",
                "message": f"请求处理异常：{str(e)}"
            }
            self.ask_response.emit(error)

    def _handle_ask_response(self, response):
        """主线程处理响应"""
        try:
            # 移除"正在生成回复"提示
            if hasattr(self, 'waiting_widget') and self.waiting_widget:
                self.chat_layout.removeWidget(self.waiting_widget)
                self.waiting_widget.deleteLater()
                self.waiting_widget = None

            self.is_generating = False  # 重置生成状态
            self.send_btn.setEnabled(True)  # 启用发送按钮

            if response.get("status") == "success":
                answer = response.get("message", "AI返回空内容")
                self._add_message_safely(f"AI回复：{answer}", True)
            else:
                error = response.get("message", "未知错误")
                self._add_message_safely(f"错误：{error}", True)

        except Exception as e:
            print(f"处理响应错误: {str(e)}")
            self._add_message_safely(f"处理响应时出错: {str(e)}", True)
        finally:
            # 确保状态重置
            self.is_generating = False
            self.send_btn.setEnabled(True)
            # 自动滚动到底部
            QTimer.singleShot(100, self._scroll_to_bottom)

    # ==================== 核心功能 ====================
    def browse_project(self):
        if path := QFileDialog.getExistingDirectory(self, "选择项目目录"):
            self.project_path.setText(path)

    def load_project(self):
        self.current_project = None
        self.current_file = None
        self.editor.clear()
        self.save_btn.setEnabled(False)
        self.current_file_label.setText("当前文件：未打开")
        raw_path = self.project_path.text()
        if not raw_path:
            self.update_status.emit("请输入有效路径", "#FF9800")
            return

        try:
            path = str(Path(raw_path).resolve())
            if not Path(path).exists():
                self.update_status.emit("路径不存在"), 
                return
        except Exception as e:
            self.update_status.emit(f"路径解析错误: {str(e)}", "#F44336")
            return

        # 先在主线程初始化模型
        self.update_status.emit("初始化文件系统...", "#2196F3")
        self.model = QFileSystemModel()
        self.model.setRootPath("")  # 先设置为空路径
        self.file_tree.setModel(self.model)

        # 使用QTimer延迟加载，避免直接在新线程中操作UI
        QTimer.singleShot(100, lambda: self._async_load_project(path))

    def _async_load_project(self, path):
        def worker():
            try:
                result = self.client.open_project(path)

                # 使用信号更新UI
                if "open_status" in result and result["open_status"]["status"] == "success":
                    self.update_file_tree.emit(path)
                    doc_msg = result["doc_status"]["data"]["message"]
                    self.add_chat_message.emit(f"文档状态：{doc_msg}", True)
                    self.update_status.emit("项目加载成功", "#4CAF50")
                    self.load_chat_history()
                else:
                    error = result.get("message", "未知错误")
                    self.update_status.emit(f"加载失败：{error}", "#F44336")

            except Exception as e:
                self.update_status.emit(f"加载异常：{str(e)}", "#F44336")

        threading.Thread(target=worker, daemon=True).start()

    def save_file(self):
        if not self.current_file:
            self.update_status.emit("请先打开文件", "#FF9800")
            return

        try:
            content = self.editor.text()
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.update_status.emit(f"保存成功: {self.current_file}", "#4CAF50")
            self.save_btn.setStyleSheet("""QPushButton { background: #388e3c; }""")
            self.save_btn.setText("保存文件")

            # 触发知识库更新
            def async_update():
                result = self.client.save_and_update()
                if result.get("status") == "success":
                    self.update_status.emit("知识库更新完成", "#4CAF50")

            threading.Thread(target=async_update, daemon=True).start()
        except Exception as e:
            self.update_status.emit(f"保存失败：{str(e)}", "#F44336")

    def handle_chat(self):
        question = self.chat_input.text().strip()
        if not question:
            return

        # 主线程添加用户消息
        self._add_message_safely(f"▶ 用户提问：{question}", False)
        self.chat_input.clear()

        def async_ask():
            try:
                # 通过信号添加AI等待提示
                self._add_message_safely("▷ AI正在思考...", True)

                response = self.client.ask_question(question)

                # 主线程处理响应
                if response.get("status") == "success":
                    self._add_message_safely(f"✓ AI回复：{response['message']}", True)
                else:
                    error = f"错误：{response.get('message', '未知错误')}"
                    self._add_message_safely(error, True)

            except Exception as e:
                error_msg = f"✗ 请求异常：{str(e)}"
                self._add_message_safely(error_msg, True)

            finally:
                # 滚动到底部
                QTimer.singleShot(100, self._scroll_to_bottom)

        threading.Thread(target=async_ask, daemon=True).start()

    def _add_message_safely(self, text, is_ai):
        """线程安全的添加消息方法"""
        self.add_chat_message.emit(text, is_ai)

    # 安全滚动到底部
    def _scroll_to_bottom(self):
        """安全滚动到底部"""
        # 确保滚动条已更新
        self.chat_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.chat_scroll.verticalScrollBar().setValue(
                self.chat_scroll.verticalScrollBar().maximum()
            )
        )

        # 立即尝试滚动
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def generate_commit_msg(self):
        def async_generate():
            try:
                self.update_status.emit("正在生成提交信息...", "#2196F3")
                result = self.client.generate_commit_message()

                if result.get("status") == "success":
                    msg = result.get("message", "")
                    self.commit_edit.setPlainText(msg)
                    self.update_status.emit("提交信息生成成功", "#4CAF50")
                else:
                    error = result.get("message", "生成失败")
                    self.update_status.emit(f"生成错误: {error}", "#F44336")

            except Exception as e:
                self.update_status.emit(f"生成异常: {str(e)}", "#F44336")

        if not self.current_project:
            self.update_status.emit("请先加载项目", "#FF9800")
            return

        threading.Thread(target=async_generate, daemon=True).start()

    def execute_commit(self):
        msg = self.commit_edit.toPlainText().strip()
        if not msg:
            self.update_status.emit("提交信息不能为空", "#FF9800")
            return

        def async_commit():
            try:
                self.update_status.emit("正在提交代码...", "#2196F3")
                result = self.client.perform_git_commit(msg)

                if result.get("status") == "提交成功":
                    self.update_status.emit("代码提交成功", "#4CAF50")
                    self.commit_edit.clear()
                else:
                    error = result.get("error", "提交失败")
                    self.update_status.emit(f"提交错误: {error}", "#F44336")

            except Exception as e:
                self.update_status.emit(f"提交异常: {str(e)}", "#F44336")

        threading.Thread(target=async_commit, daemon=True).start()

    # ==================== 信号处理 ====================
    def handle_status_update(self, text, color):
        self.statusBar().showMessage("")
        label = QLabel(text)
        label.setStyleSheet(f"background-color: {color}; color: white; padding: 8px;")
        self.statusBar().addWidget(label)
        QTimer.singleShot(3000, lambda: self.statusBar().removeWidget(label))

    def handle_new_message(self, text, is_ai):
        msg = ChatMessage(is_ai, markdown_text=text)
        self.chat_layout.addWidget(msg)  # 改为追加到末尾

        # 确保消息被正确渲染后再滚动
        QTimer.singleShot(50, self._scroll_to_bottom)

        # 添加额外的滚动保证
        QTimer.singleShot(200, self._scroll_to_bottom)

    def handle_file_tree_update(self, path):
        # 确保在主线程执行
        try:
            if hasattr(self, 'model'):
                self.model.setRootPath(path)
                root_index = self.model.index(path)
                self.file_tree.setRootIndex(root_index)
                self.current_project = path

                # 设置合适的列宽
                self.file_tree.setColumnWidth(0, 300)
                for i in range(1, self.model.columnCount()):
                    self.file_tree.setColumnHidden(i, True)
        except Exception as e:
            self.update_status.emit(f"文件树更新失败: {str(e)}", "#F44336")

    def open_file(self, index):
        try:
            self.current_file = self.model.filePath(index)
            with open(self.current_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.editor.setText(content)

                # 启用保存按钮
                self.save_btn.setEnabled(True)

                # 更新文件标签（修复显示问题）
                self.current_file_label.setText(f"当前文件: {self.current_file}")
                self.update_status.emit(f"已打开文件：{self.current_file}", "#2196F3")

                # 连接修改检测
                self.editor.textChanged.connect(self.on_editor_content_changed)

        except Exception as e:
            self.save_btn.setEnabled(False)
            self.update_status.emit(f"打开失败：{str(e)}", "#F44336")

    def on_editor_content_changed(self):
        """当编辑器内容变化时更新保存按钮状态"""
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #ff9800;
                color: white;
            }
            QPushButton:hover {
                background: #fb8c00;
            }
        """)
        self.save_btn.setText("保存*")

    def load_chat_history(self):
        self.on_history_loaded()

        # 移除所有弹性空间
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.spacerItem():  # 关键：移除所有弹性占位
                self.chat_layout.removeItem(item)

        # 添加加载提示（保证在最顶部）
        try:
            loading_msg = ChatMessage(is_ai=True, markdown_text="┌ 加载历史记录中...")
            self.chat_layout.addWidget(loading_msg)
            QTimer.singleShot(100, self._scroll_to_bottom)
        except Exception as e:
            print(f"加载提示异常: {str(e)}")

        def async_load():
            try:
                resp = requests.get("http://127.0.0.1:5000/get_chat_history", timeout=240)
                if resp.status_code == 200:
                    history = resp.json().get("history", [])
                    # 按时间顺序加载（旧消息在前）
                    for msg in history:
                        content = f"{msg['role']}:{msg['content']}"
                        is_ai = msg.get("role") == "bot"
                        self.add_chat_message.emit(content, is_ai)
                    QTimer.singleShot(100, self._scroll_to_bottom)
            except Exception as e:
                error = f"加载历史记录失败：{str(e)}"
                self.add_chat_message.emit(error, True)

        threading.Thread(target=async_load, daemon=True).start()  # 移出函数定义

    # 添加克隆仓库的方法
    def clone_repository(self):
        """打开克隆仓库对话框并处理克隆操作"""
        dialog = CloneDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            repo_url = data["repo_url"]
            save_path = data["save_path"]

            if not repo_url or not save_path:
                self.update_status.emit("请填写完整的仓库URL和保存路径", "#FF9800")
                return

            # 显示加载状态
            self.update_status.emit(f"正在克隆仓库: {repo_url}...", "#2196F3")

            # 在后台线程执行克隆操作
            threading.Thread(
                target=self._async_clone_repository,
                args=(repo_url, save_path),
                daemon=True
            ).start()

    def _async_clone_repository(self, repo_url, save_path):
        """异步执行克隆操作"""
        try:
            result = self.client.clone_repository(repo_url, save_path)

            if result.get("status") == "success":
                # 克隆成功后自动加载项目
                self.project_path.setText(save_path)
                QTimer.singleShot(1000, self.load_project)  # 延迟1秒加载
                self.update_status.emit("仓库克隆成功!", "#4CAF50")
            else:
                error = result.get("message", "克隆失败")
                self.update_status.emit(f"克隆错误: {error}", "#F44336")

        except Exception as e:
            self.update_status.emit(f"克隆异常: {str(e)}", "#F44336")

class ChatAddEvent(QEvent):
    TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, content, is_ai):
        super().__init__(self.TYPE)
        self.content = content
        self.is_ai = is_ai

        # 重写事件处理
        def event(self, event):
            """处理自定义事件"""
            if isinstance(event, IDEWindow.ChatAddEvent):
                msg = ChatMessage(event.is_ai, event.content)
                self.chat_layout.addWidget(msg)
                return True
            return super().event(event)

        # 强制UI刷新
        def _force_ui_refresh(self):
            self.chat_content.updateGeometry()
            self.chat_scroll.viewport().update()
            self._scroll_to_bottom()
        #
        # threading.Thread(target=async_load, daemon=True).start()

class CloneDialog(QDialog):
    """克隆仓库对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("克隆Git仓库")
        self.setFixedSize(500, 200)

        layout = QVBoxLayout()

        # 仓库URL输入
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("仓库URL:"))
        self.repo_url = QLineEdit()
        self.repo_url.setPlaceholderText("https://github.com/username/repository.git")
        url_layout.addWidget(self.repo_url)
        layout.addLayout(url_layout)

        # 保存路径输入
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("保存路径:"))
        self.save_path = QLineEdit()
        self.save_path.setPlaceholderText("选择或输入保存位置...")
        path_layout.addWidget(self.save_path)

        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_save_path)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # 按钮区域
        btn_layout = QHBoxLayout()
        clone_btn = QPushButton("克隆")
        clone_btn.setStyleSheet("background: #4CAF50; color: white;")
        clone_btn.clicked.connect(self.accept)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("background: #f44336; color: white;")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(clone_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_save_path(self):
        """选择保存路径"""
        path = QFileDialog.getExistingDirectory(self, "选择保存位置")
        if path:
            self.save_path.setText(path)

    def get_data(self):
        """获取表单数据"""
        return {
            "repo_url": self.repo_url.text().strip(),
            "save_path": self.save_path.text().strip()
        }

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = IDEWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序崩溃: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def __init__(self):
    # 确保所有信号槽连接
    self._connect_signals()

def _connect_signals(self):
    """确保所有信号槽连接"""
    # 保存按钮
    self.save_btn.clicked.connect(self.save_file)
