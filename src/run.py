from multiprocessing.connection import answer_challenge

from exceptiongroup import catch
from flask import Flask, request, render_template, jsonify, current_app, Response, stream_with_context
import subprocess
import json
import faiss
import time
import os
from flask_sqlalchemy import SQLAlchemy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# 文本分割工具导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
from datetime import datetime
from langchain_core.documents import Document
import re
from langchain_ollama import OllamaLLM
from pythonFile.PDFToJSON import pdf_to_md
from pythonFile.getRagFileList import get_ragfile_list_method
# 线程锁，用来优化第一次部署的时间
from threading import Lock
# 异步任务处理（用于解决提问时调用大语言模型搜素速度太慢，最大线程：4）
from concurrent.futures import ThreadPoolExecutor
import random, re
from langchain_core.runnables import RunnablePassthrough

executor = ThreadPoolExecutor(max_workers=2)
# 用于解决，新线程无法自动继承 Flask 的应用上下文导致ask方法调用configure_llm_chain方法的数据库时会出错
from flask import copy_current_request_context
# git转md
from pythonFile.gitToMd import *
# 代码差异对比
import difflib  # 用于代码差异对比
# 模型缓存单例
_model_cache = {}
_cache_lock = Lock()

app = Flask(__name__)  # 创建Flask应用实例
# 配置问答大语言模型
def get_llm():
    with _cache_lock:
        if "llm" not in _model_cache:
            _model_cache["llm"] = OllamaLLM(
                model="deepseek-r1:1.5b",
                temperature=0.3,  # 降低随机性
                num_ctx=4096,  # 限制上下文窗口
                num_predict=2048,
            )
        return _model_cache["llm"]

# 配置知识库调用的模型(自然语言处理)
def get_embeddings():
    if "embeddings" in _model_cache:
        return _model_cache["embeddings"]

    try:
        import torch
        # 自动检测GPU可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[设备检测] 使用设备: {device}")

        # 显式指定模型名称
        model_name = "BAAI/bge-small-zh-v1.5"

        # 添加模型缓存路径设置
        cache_folder = Path.home() / ".cache/huggingface"
        cache_folder.mkdir(parents=True, exist_ok=True)

        # 初始化模型
        _model_cache["embeddings"] = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(cache_folder),  # 显式指定缓存路径
            model_kwargs={"device": device},
            encode_kwargs={
                'batch_size': 8,
                'normalize_embeddings': True
            }
        )

        # 验证模型加载
        test_text = "验证嵌入模型"
        embedding = _model_cache["embeddings"].embed_query(test_text)
        print(f"[模型验证] 测试嵌入维度: {len(embedding)}")

        return _model_cache["embeddings"]

    except Exception as e:
        print(f"[严重错误] 嵌入模型初始化失败: {str(e)}")
        raise RuntimeError(f"无法加载嵌入模型: {str(e)}") from e

# 全局变量改为字典，按 chatNum 存储
documents_dict = {}  # 存储每个对话的文档
vectorstore_dict = {}  # 存储每个目录的向量数据库

# 临时的文本存储,确保页面上正在编辑的ma文件内容与这里存储的一样
temp_context = ""

# QA_chain缓存
qa_chain_cache = {}
_qa_cache_lock = Lock()

# 修改后的FAISS配置
from faiss import StandardGpuResources

gpu_res = StandardGpuResources()

from flask_sqlalchemy import SQLAlchemy

# 初始化数据库
db = SQLAlchemy()

# 去除deepseek中的思考内容，降低回复冗余度
def remove_think_tags(response):
    response = re.sub(
        r'<think>.*?</think>',
        '',
        response,
        flags=re.DOTALL  # 支持跨行匹配
    )
    return response

def read_file_with_fallback(file_path: Path) -> str:
    """安全读取含多种编码的文件"""
    encodings = ['utf-8', 'gbk', 'latin-1']
    raw_data = file_path.read_bytes()

    for encoding in encodings:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw_data.decode('utf-8', errors='replace')

# 改进的aider核心功能
import shutil

# 调用aider
def aider_simple(instruction: str) -> str:
    global output
    tmp_path = None
    print(f"[DEBUG]aider_simple得到的内容是:{instruction}]")
    """修复后的安全代码执行函数"""
    try:
        import sys
        # 获取aider绝对路径
        conda_env_path = Path(sys.executable).parent  # 重要：Anaconda路径结构不同
        aider_path = conda_env_path / "Scripts/aider.exe"
        print(f"[DEBUG] 预期aider路径: {aider_path}")  # 添加调试输出
        if not aider_path.exists():
            return "错误：aider未安装，请执行 'pip install aider-chat'"
        # 转换为绝对路径并验证

        # # 创建临时副本
        # with tempfile.NamedTemporaryFile(
        #         mode="w",
        #         encoding="utf-8",
        #         suffix=target_path.suffix,
        #         delete=False
        # ) as tmp:
        #     orig_content = read_file_with_fallback(target_path)
        #     tmp.write(orig_content)
        #     tmp_path = tmp.name
        #     print(f"[DEBUG] 临时文件路径: {tmp_path}")

        # 执行aider命令
        cmd = [
            # aider所在位置
            str(aider_path),
            "--yes",
            "--no-show-model-warnings",
            "--no-auto-lint",  # 禁用自动代码检查
            "--no-auto-test",  # 禁用自动测试
            "--message", f'"{instruction}"'
        ]
        print(f"[DEBUG] 执行命令: {cmd}")
        process = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            encoding="utf-8",
            cwd=str(get_project_root())
        )

        get_branch_cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        branch_process = subprocess.run(
            get_branch_cmd,
            text=True,
            capture_output=True,
            encoding="utf-8",
            cwd=str(get_project_root())
        )

        if branch_process.returncode == 0:
            branch_name = branch_process.stdout.strip()
            print(f"[DEBUG] 当前分支: {branch_name}")

            # 执行git push
            push_cmd = ["git", "push", "origin", branch_name]
            push_process = subprocess.run(
                push_cmd,
                text=True,
                capture_output=True,
                encoding="utf-8",
                cwd=str(get_project_root())
            )

            # 输出推送结果
            if push_process.returncode == 0:
                print(f"[SUCCESS] 成功推送分支 {branch_name} 到远程仓库")
                print(push_process.stdout)
            else:
                print(f"[ERROR] 推送失败: {push_process.stderr}")
        else:
            print(f"[ERROR] 无法获取当前分支: {branch_process.stderr}")

        # 修改返回处理部分
        output = process.stdout + process.stderr
        print(f"[DEBUG] aider输出: {output}")
    except Exception as e:
        print(f"[WARN]aider_simple警告:{str(e)}")
    finally:
        try:
            prompt = f"""
            下文是大概率是一段有可能是中文或者英文的内容，由用户调用其他大语言模型生成的产物
            你根据这段内容分析或者翻译，总结成下面3点
            仅对中文字段进行分析，英文字段以及不涉及到AI回答的内容全部无视
            1：AI的回复内容
            2：对什么文件进行了修改或提问
            3：是否成功更改了代码
            上下文{output}
            回答的内容为必须包含上述三点的一段话，至多100字。
            """
            fin_result = get_llm().invoke(prompt).lower()
            fin_result = remove_think_tags(fin_result)
            print(f"[DEBUG]aider最终回答：: {fin_result}")
            return str(f"自动通过aider进行代码修改,修改内容总结如下"
                       f"\n{fin_result}\n"
                       f"\n若需要详细的aider输出信息,请查阅您项目中的.aider.chat.history.md文件")
        except Exception as e:
            return f"修改代码失败，请用户自行进行代码修改,错误内容:{str(e)}"


# 生成目录树（示例输出）
def generate_dir_tree(path, indent=0):
    tree = []
    for item in Path(path).iterdir():
        # 忽略隐藏文件和特定目录
        if item.name.startswith(('.', '__')):
            continue

        prefix = "│   " * indent + "├── "
        if item.is_dir():
            tree.append(f"{prefix}文件夹:{item.name}/")
            tree.extend(generate_dir_tree(item, indent + 1))
        else:
            tree.append(f"{prefix}文件{item.name}")
    return tree


import sqlite3
import os

# 获取数据库
def get_db(project_root):
    """获取数据库连接，自动创建数据库文件"""
    db_path = os.path.join(project_root, "chat_history.db")
    conn = sqlite3.connect(db_path)

    # 创建表（如果不存在）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            roler TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    return conn

def save_conversation(project_root, message, roler):
    """保存对话记录"""
    try:
        conn = get_db(project_root)
        conn.execute('''
            INSERT INTO conversations (message, roler) 
            VALUES (?, ?)
        ''', (message, roler))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"保存对话失败: {str(e)}")


def get_conversation(project_root):
    """获取对话记录（仅修改这里）"""
    try:
        conn = get_db(project_root)
        cursor = conn.cursor()  # 创建游标

        # 执行查询
        cursor.execute('SELECT * FROM conversations')
        results = cursor.fetchall()  # 获取所有结果

        conn.close()
        return results  # 返回查询结果

    except Exception as e:
        print(f"获取对话失败: {str(e)}")
        return []


@app.route("/test_retrieval", methods=["POST"])
def test_retrieval():
    global extract_keywords
    try:
        # 强制获取请求数据
        query = request.json.get("question", "")
        if not query.strip():
            return jsonify({"error": "问题内容不能为空"}), 400

        # 加载向量库（增加空值检查）
        vectorstore = load_vectorstore()
        if vectorstore is None:
            return jsonify({"error": "知识库初始化失败，请先构建知识库"}), 500

        # 创建检索器
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "fetch_k": 50, "lambda_mult": 0.5}
        )

        # 执行检索（增加超时处理）
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            return jsonify({"error": f"检索过程失败: {str(e)}"}), 500

        # 处理空结果
        if not docs:
            return jsonify({
                "query": query,
                "warning": "未找到相关文档",
                "retrieved_docs": []
            })

        # 处理文档数据
        result = []
        for doc in docs:
            try:
                # 安全获取元数据
                source = doc.metadata.get('source', '未知来源')
                score = doc.metadata.get('score', 0.0)

                # 提取关键词
                def extract_keywords(text):
                    words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
                    return list(set(words))

                # 构建结果
                result.append({
                    "doc_keywords": extract_keywords(doc.page_content),
                    "match_rate": round(
                        len(set(query.split()) & set(extract_keywords(doc.page_content))) / len(query.split()), 2),
                    "source": str(Path(source).resolve()),  # 返回绝对路径
                    "score": float(score),
                    "metadata": doc.metadata
                })
            except Exception as doc_error:
                print(f"文档处理异常: {str(doc_error)}")
                continue  # 跳过错误文档

        return jsonify({
            "query": query,
            "query_keywords": extract_keywords(query),
            "retrieved_docs": result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "服务器内部错误",
            "detail": str(e)
        }), 500

# 此路由将在用户打开项目目录时自动调用，目的就是生成一个md文件
# 无输入，输出生成成功提示
# 过程：生成一个项目文档
# 备注，暂时不对名字进行更改
@app.route("/askDemo")
def askDemo():
    if (get_doc_path()/"doc.md").exists():
        # 修改 /askDemo 中已存在文档的返回结构
        return jsonify({
            "status": "success",
            "data": {  # 添加 data 字段统一包装业务数据
                "message": "已经生成过文档，不再自动生成",
                "doc_content": None  # 明确返回关键字段
            }
        })
    try:
        print("\n[DEBUG]未检测到项目文档，正在准备生成")
        # 获取目录
        dir_tree = "\n".join(generate_dir_tree(get_project_root())[:100])
        # 获取COMMIT_HISTORY中的git提交记录
        file_path = Path(get_doc_path()/"COMMIT_HISTORY.md")
        git_log = None
        if file_path.exists():
            # 一次性读取内容
            git_log = file_path.read_text(encoding="utf-8")
            print(git_log)
        else:
            print(f"文件 {file_path} 不存在")

        design_chain = (
                {
                    "git_log": lambda x: git_log,
                    "dir_tree": lambda x: dir_tree
                }
                | PromptTemplate.from_template(f"""
                    你是一个软件项目工程师，你即将对下面目录呈现的项目进行文档生成的操作，目录如下：
                    {dir_tree}
                    对应的部分git信息如下：
                    {git_log}
                    为我写一个项目的基于markdown的设计文档，不要混杂任何的HTML语法我会给你一个模板，然后要求如下。
                    1：“[]”中的内容由你填写，提示就在符号内。
                    2：“<>()”是我定义的特殊符号，意思为把<>中的内容按照()中的提示再多写若干个，直到达到要求。
                    模板如下：
                    # [项目名称，如果上下文没有明显的项目名称，由你取名]
                    ## 概要设计：[用一段话来描述整个项目]
                    ## 项目功能：
                    <### 功能点1：[功能名称]
                        [输入]
                        [输出]>(按照目录层级去推测整个项目可能具有的功能，以及相应的输出，并顺序罗列成功能点格式)
                    ## 设计
                    ### [以你的方式去书写这个文档的设计，要结合整个项目进行分析，但至少要考虑采用的语言，数据库，用例图]
                    ## 测试用例
                    <### 功能点1-测试用例1:
                        [测试内容]
                        [测试方式]
                        [预计输入、输出]
                        测试结果：待定
                    >（以你上面生成的多个功能点来为他们写若干个测试用例，编号名就如“功能点1-测试用例1”,“功能点2-测试用例1”这一类进行分点，每个测试用例要有测试方式，预计输入输出）
                  """)
                | get_llm()
                | StrOutputParser()
        )
        # 获得文档
        result = design_chain.invoke({})
        result = remove_think_tags(result)
        # 假设result是字符串类型，如果不是需要先转换
        result_str = str(result)
        output_path = os.path.join(get_doc_path(), 'doc.md')
        # 将文档写入md文件中，并且因为使用的是deepseek模型，需要把思考的内容去除
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_str)
        print(f"[DEBUG]生成完成，请从doc文件夹中打开并查看")
        # 更新知识库，把知识库中的内容保存
        update_dataBase()

        return jsonify({
            "status": "success",
            "message": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"处理请求失败: {str(e)}"
        }), 500


# RAG向量知识库更新框架
# 流程：1：遍历全项目，获取文件最近一次更新时间和缓存更新时间对比（初始化项目的时候就是第一缓存更新的时间）
# 2:时间对不上的话就记录下来，直到遍历完所有文件为止
# 3：删除知识库中的这些对不上时间的文件，再写入更新后的文件（也要切分等）
import os
from pathlib import Path
from datetime import datetime

# 全局缓存字典：记录每个文件的最后修改时间
# 结构：{ "文件路径1": 时间戳1, "文件路径2": 时间戳2 }
FILE_MTIMES = {}

def check_file_modified(file_path: Path) -> bool:
    """使用Path对象"""
    try:
        current_mtime = file_path.stat().st_mtime
        cached_mtime = FILE_MTIMES.get(str(file_path))
        return current_mtime != cached_mtime
    except Exception as e:
        print(f"文件检查失败: {file_path} - {str(e)}")
        return False

def format_time(timestamp):
    """将时间戳转换为可读格式"""
    if timestamp is None:
        return "从未记录"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def should_skip(file_path: Path) -> bool:
    skip_dirs = {'.git', 'vectorstore', '__pycache__'}
    skip_suffix = {'.pyc', '.pyo', '.pyd', '.pkl','.db'}

    # 检查父目录
    if any(part in skip_dirs for part in file_path.parts):
        return True

    return file_path.suffix in skip_suffix

def load_file(file_path):
    """根据文件类型加载内容"""
    # 取后缀对代码进行微调
    ext = Path(file_path).suffix
    try:
        # 步骤1：跳过一写没法读取或者不需要读取的非代码或文档文件
        if should_skip(path):
            return ""
        # 文本文件直接读取
        if ext in ['.md', '.txt']:
            return open(file_path, 'r', encoding='utf-8').read()

        # 代码文件保留注释
        elif ext in ['.py', '.java']:
            return open(file_path, 'r', encoding='utf-8').read()

    except Exception as e:
        raise RuntimeError(f"文件读取失败: {str(e)}")


def split_documents(content, file_path):
    """
    分割文档内容并添加完整元数据
    特别优化Python代码文件的分割
    """
    try:
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
        file_type = file_path.suffix.lower()

        # 获取文件元数据
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        file_name = file_path.name

        # 1. 特别处理Python代码文件
        if file_type == '.py':
            return process_python_file(content, file_path, file_mtime)

        # 2. 其他文件处理（保持原逻辑）
        return process_other_file(content, file_path, file_type, file_mtime)

    except Exception as e:
        print(f"文档分割失败：{file_path} - {str(e)}")
        return []

# 处理非代码文件的标准办法
def process_other_file(content, file_path, file_type, file_mtime):
    file_name = file_path.name
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "；", "?", "!", ". ", "```"]
    )

    # 在内容前添加文件名标记
    marked_content = f"【File:{file_name}】\n{content}"

    docs = text_splitter.create_documents(
        texts=[marked_content],
        metadatas=[{
            "source": str(file_path.resolve()).lower().replace('\\', '/'),
            "last_modified": file_mtime,
            "file_type": file_type,
            "file_size": os.path.getsize(file_path)
        }]
    )

    # 增强元数据
    for doc in docs:
        doc.metadata.update({
            "page": docs.index(doc) + 1,  # 块序号
            "total_pages": len(docs)  # 总块数
        })

    print(f"成功分割文件：{file_path} → {len(docs)}个块")
    return docs

# 处理python文件的标准方法（class分割）
def process_python_file(content, file_path, file_mtime):
    """针对Python代码文件的优化分割方法"""
    file_name = file_path.name

    # 在内容前添加文件名标记
    marked_content = f"【File:{file_name}】\n{content}"

    # 使用专门为Python代码设计的分割器
    python_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",  # 空行分割
            "\nclass ",  # 类定义
            "\ndef ",  # 函数定义
            "\n# ",  # 注释分割
            "\n```"  # 代码块结束
        ],
        keep_separator=True  # 保留分隔符使代码结构完整
    )

    docs = python_splitter.create_documents(
        texts=[marked_content],
        metadatas=[{
            "source": str(file_path.resolve()).lower().replace('\\', '/'),
            "last_modified": file_mtime,
            "file_type": ".py",
            "file_size": os.path.getsize(file_path)
        }]
    )

    # 增强元数据 - 识别代码块类型
    for doc in docs:
        content = doc.page_content

        # 自动识别块类型
        block_type = "code"
        if "\nclass " in content:
            block_type = "class"
        elif "\ndef " in content:
            block_type = "function"
        elif content.strip().startswith("#"):
            block_type = "comment"

        doc.metadata.update({
            "page": docs.index(doc) + 1,  # 块序号
            "total_pages": len(docs),  # 总块数
            "block_type": block_type  # 新增：块类型
        })

    print(f"成功分割Python文件：{file_path} → {len(docs)}个块")
    return docs

import threading

# 全局化用户已经打开了的项目目录，方便后端调用
config = {
    "VECTORSTORE_PATH": Path("pythonFile/vectorstore").resolve(),
    "PROJECT_ROOT": Path("pythonFile/src").resolve(),
    "DOC_PATH": Path("pythonFile/doc").resolve(),
    "lock": threading.Lock()  # 添加线程锁保证线程安全
}

# embeddings = get_embeddings()

# 打开一个项目
# 输入：文档路径
# 过程：更新或获取知识库
# 输出：过程结果
@app.route('/openFilePath', methods=['POST'])
def openFilePath():
    try:
        data = request.get_json(force=True)
        project_dir = Path(data['file_path']).resolve()
        # 其他初始化
        with config["lock"]:
            config["PROJECT_ROOT"] = project_dir / "src"
            config["VECTORSTORE_PATH"] = project_dir / "vectorstore"
            config["DOC_PATH"] = project_dir / "doc"

        os.chdir(project_dir)
        gitToMdFile(project_dir)
        update_dataBase()
        return jsonify({"status": "success", "message": f"项目初始化成功，数据库位于 {project_dir}"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

from pathlib import Path

# 从前端获取本地向量数据库路径和项目文件夹路径
def get_vectorstore_path() -> Path:
    """线程安全获取向量库路径"""
    with config["lock"]:
        return config["VECTORSTORE_PATH"]  # 返回副本避免引用问题

def get_project_root() -> Path:
    """线程安全获取项目根路径"""
    with config["lock"]:
        path = config["PROJECT_ROOT"]
        if not path.exists():
            raise FileNotFoundError(f"项目目录不存在: {path}")
        return path

def get_doc_path() -> Path:
    """线程安全获取文档目录"""
    with config["lock"]:
        path = config["DOC_PATH"]
        path.mkdir(parents=True, exist_ok=True)  # 自动创建目录
        return path
# 加载向量数据库
def load_vectorstore() -> FAISS | None:
    """加载失败时自动初始化空库"""
    try:
        vs = FAISS.load_local(
            folder_path=str(get_vectorstore_path()),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True
        )
        return vs
    except Exception as e:
        print(f"首次加载失败，自动创建空库: {str(e)}")
        empty_store = FAISS.from_texts([], get_embeddings())
        empty_store.save_local(str(get_vectorstore_path()))
        return empty_store

import subprocess
import tempfile
import shutil
from pathlib import Path

# 在问答处理前添加意图判断
def detect_code_intent(query: str) -> bool:
    prompt =  f"""
        # 代码修改需求判断指令
        
        ## 任务要求
        严格分析用户问题，判断是否需要修改代码文件（如.py/.java/.js等源码），按以下逻辑输出结果：
        如果需要，就输出YES，否则输出NO
        且一般来说，只要用户的提问中携带“代码二字”，都视为YES
               是 → YES    否 → NO
        
        2. **输出规范**
        - 仅允许输出YES/NO（字母大写）
        - 依据必须来自问题的显性表述，禁止推测
        
        ## 当前问题分析
        用户问题：{query}
        
        ## 禁止推断场景
        × 文档更新需求（如：修改README.md）
        × 配置文件调整（如：修改config.yaml）
        × 数据内容变更（如：更新测试数据）
        × 非代码文件操作（如：图片/音频处理）
        
        ## 示例参考
        [问题] "优化UserController.java的响应速度"
        [分析] 包含代码文件+修改动词 → YES
        
        [问题] "帮我查API文档"
        [分析] 无代码文件或修改动作 → NO
        
        ## 最终输出
    """
    result = get_llm().invoke(prompt).lower()
    result = remove_think_tags(result)
    print("[DEBUG]该问题是否需要代码修改："+result+"\n========================")
    return any(word in result for word in {"YES", "yes"})

# 将这些文件视为代码文件
CODE_EXTENSIONS = {'.py', '.js', '.java', '.c', '.cpp', '.html', '.css'}

def is_code_file(file_path):
    return os.path.splitext(file_path)[1].lower() in CODE_EXTENSIONS

# 用户提问，如果是普通的问题提问，正常回答；如果是带有代码修改性质的提问，问确认是否要更改代码
@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json["question"]
        app.logger.info(f"[用户提问] {query}")

        # 第一阶段：首先判断是否为简单问题（不涉及代码/项目）
        if is_simple_query(query):
            # 直接处理简单问题，不加载知识库
            simple_response = handle_simple_query(query)
            save_conversation(get_project_root(), simple_response, "bot")
            return jsonify({
                "status": "success",
                "message": simple_response,
                "timestamp": datetime.now().isoformat()
            })

        # 第二阶段：需要知识库的问题处理
        vectorstore = load_vectorstore()
        if not vectorstore:
            return jsonify({"error": "向量库加载失败"}), 500

        # 检索策略
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20
            }
        )
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            # 如果失败，回退到旧方法
            docs = retriever.get_relevant_documents(query)
            print(f"使用旧方法获取文档: {str(e)}")

        # 构建上下文时过滤低相关度内容
        rag_context = ""
        if docs:
            # 打印每个文档的元数据
            print(f"[DEBUG] 检索到的文档数量: {len(docs)}")

            # 输出每个文档的关键信息
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', '未知来源')
                content_preview = doc.page_content[:50] + '...' if len(doc.page_content) > 50 else doc.page_content

                print(f"  [文档 {i + 1} 元数据]")
                print(f"    来源: {source}")
                print(f"    文件类型: {doc.metadata.get('file_type', '未知')}")
                print(f"    页码: {doc.metadata.get('page', '无')}/{doc.metadata.get('total_pages', '无')}")
                print(f"    相关性分数: {getattr(doc, 'metadata_score', '未记录')}")
                print(f"    内容预览: {content_preview}")
                print(f"    内容长度: {len(doc.page_content)} 字符")
                print()

            # 构建RAG上下文
            rag_context = "\n\n".join([
                f"=== 相关片段 {i + 1} ===\n{doc.page_content}"
                for i, doc in enumerate(docs[:5])
            ])

            # 打印最终的RAG上下文信息
            print(f"[DEBUG] 构建的RAG上下文:")
            print(f"  包含文档片段数: {len(docs[:5])}")
            print(f"  上下文总长度: {len(rag_context)} 字符")
            print(f"  上下文前200字符: {rag_context[:200]}...\n")
        else:
            print("[DEBUG] 未检索到相关文档")
            rag_context = "没有找到相关背景信息"

        save_conversation(get_project_root(), query, "user")

        # 正常提问提示词模板
        prompt_template = f"""
            请根据用户的问题提供直接的回答。如果你无法从以下相关内容中找到答案，请明确告知：
            用户问题：{query}
            
            相关背景信息：
            {rag_context if rag_context else "没有找到相关背景信息"}
            
            请遵循以下规则：
            1. 直接回答问题，不添加额外解释
            2. 如果无法从背景信息中获取答案，应明确说明
            3. 避免分析问题类型或回答内容的意义
        """

        # 第三阶段：判断是否为代码修改请求
        if detect_code_intent(query):
            # 调用Aider进行代码修改
            message = aider_simple(query)
            save_conversation(get_project_root(), message, "bot")
            update_dataBase()
            return jsonify({
                "status": "success",
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # 普通知识库问答
            response = get_llm().invoke(prompt_template)
            response = remove_think_tags(response)
            save_conversation(get_project_root(), response, "bot")
            return jsonify({
                "status": "success",
                "message": response,
                "timestamp": datetime.now().isoformat()
            })

    except Exception as e:
        app.logger.error(f"问答失败: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"处理失败: {str(e)}"
        }), 500

# === 新增辅助函数 ===
def is_simple_query(query):
    """检测是否为简单问答不需要知识库"""
    # 检测常见简单问题类型
    simple_patterns = [
        r"^(你好|hello|hi|早上好|午安|晚上好)",
        r"^(谢谢|感谢|辛苦了)",
        r"^(你的名字|你是谁)",
        r"^(再见|拜拜|告辞)",
        r"^[\W\d]*$"  # 纯符号/数字
    ]

    # 检测问题长度（太短的问题可能不需要深入处理）
    short_query_threshold = 5  # 字符数阈值

    for pattern in simple_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    if len(query.strip()) <= short_query_threshold:
        return True

    return False

def handle_simple_query(query):
    """直接处理简单问候类问题"""
    # 匹配简单的问候语
    greeting_responses = {
        r"^(你好|hello|hi|早上好|午安|晚上好)": ["你好！我是代码助手，有什么可以帮您的吗？", "您好！"],
        r"^(谢谢|感谢|辛苦了)": ["不用客气！很高兴能帮到您", "这是我的职责！"],
        r"^(你的名字|你是谁)": ["我是您的AI编程助手，随时为您解答代码问题", "我是代码小助手"],
        r"^(再见|拜拜|告辞)": ["再见！有问题随时找我", "期待下次再会！"]
    }

    for pattern, responses in greeting_responses.items():
        if re.search(pattern, query, re.IGNORECASE):
            return random.choice(responses)

    # 通用简短回答
    return random.choice([
        "我在，需要帮助吗？",
        "有什么编程问题可以问我的",
        "有什么代码方面的问题可以问我哦～"
    ])

# 添加任务状态跟踪
from collections import deque
task_queue = deque(maxlen=10)

def is_safe_path(path: Path) -> bool:
    """检查是否在项目目录内"""
    return path.is_relative_to(get_project_root())
from itertools import chain  # 必须导入这个

# 新增路径标准化函数
def normalize_path(p):
    """统一路径格式：解析绝对路径 -> 统一为小写 -> 转换为POSIX格式"""
    return Path(p).resolve().as_posix().lower()

def update_vectorstore():
    """线程安全的向量库更新"""
    try:
        vs_path = get_vectorstore_path()
        proj_root = get_project_root()
        doc_root = get_doc_path()

        # === 步骤1：加载现有库 ===
        uv_vectorstore = None
        if vs_path.exists() and any(vs_path.glob("index.*")):
            try:
                uv_vectorstore = FAISS.load_local(
                    str(vs_path),
                    get_embeddings(),
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"加载库失败将重建: {e}")

        # === 步骤2：扫描所有有效文件 ===
        print(f"[DEBUG] 正在扫描目录: {proj_root}, {doc_root}")
        current_files = []
        for root_dir in [proj_root, doc_root]:
            if not root_dir.exists():
                print(f"目录不存在: {root_dir}")
                continue
            for file_path in root_dir.rglob('*'):
                if should_skip(file_path) or not file_path.is_file():
                    continue
                current_files.append(file_path.resolve())

        # === 步骤3：处理更新文件（收集修改文件的旧块ID） ===
        processed = set()
        new_docs = []
        modify_stale_ids = []  # 新增：收集因文件修改需要删除的旧块ID

        for file_path in current_files:
            norm_path = normalize_path(file_path)
            current_mtime = file_path.stat().st_mtime
            cached_mtime = FILE_MTIMES.get(norm_path, 0)

            if current_mtime != cached_mtime or norm_path not in FILE_MTIMES:
                try:
                    # 如果是修改过的文件，收集旧块源
                    if uv_vectorstore and norm_path in FILE_MTIMES:
                        for doc_id, doc in uv_vectorstore.docstore._dict.items():
                            doc_source_path = normalize_path(Path(doc.metadata["source"]))
                            if doc_source_path == norm_path:
                                modify_stale_ids.append(doc_id)

                    # 处理新内容
                    content = read_file_with_fallback(file_path)
                    chunks = split_documents(content, file_path)
                    print(f"[处理文件] {norm_path} → {len(chunks)}块")
                    new_docs.extend(chunks)
                    FILE_MTIMES[norm_path] = current_mtime
                    processed.add(norm_path)
                except Exception as e:
                    print(f"处理失败: {file_path} - {e}")

        # === 步骤4：处理删除文件（合并所有需删除的ID） ===
        if uv_vectorstore:
            # 获取向量库中所有文档的源路径
            current_paths = {normalize_path(p.resolve()) for p in current_files}

            # 1. 收集因文件删除的旧块ID
            delete_stale_ids = [
                doc_id for doc_id, doc in uv_vectorstore.docstore._dict.items()
                if normalize_path(Path(doc.metadata["source"])) not in current_paths
            ]

            # 2. 合并修改和删除的ID，并去重
            all_stale_ids = list(set(modify_stale_ids + delete_stale_ids))

            if all_stale_ids:
                print(f"[删除旧块] 总数: {len(all_stale_ids)}")
                uv_vectorstore.delete(all_stale_ids)
                uv_vectorstore.save_local(str(vs_path))
                print(f"[删除后块数] {uv_vectorstore.index.ntotal}")

        # === 步骤5：添加新块 ===
        if new_docs:
            print(f"[添加新块] 总数: {len(new_docs)}")
            if uv_vectorstore:
                uv_vectorstore.add_documents(new_docs)
            else:
                uv_vectorstore = FAISS.from_documents(new_docs, get_embeddings())
            uv_vectorstore.save_local(str(vs_path))
            print(f"[最终块数] {uv_vectorstore.index.ntotal}")

    except Exception as e:
        print(f"更新失败: {str(e)}")
        import traceback
        traceback.print_exc()

# 更新向量数据库
# 输入：无
# 过程：更新缓存中项目文件的向量数据库
# 输出: 过程结果
@app.route('/update_dataBase')
def update_dataBase():
    update_vectorstore()
    return jsonify({"status": "success"})

def get_code_diff() -> str:
    """获取git差异内容（增强编码处理）"""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            encoding='utf-8',  # 显式指定编码
            errors='replace',  # 替换解码错误字符
            check=True,
            cwd=str(get_project_root())
        )
        print(f"[DEBUG]git获取到的差异信息: {result.stdout}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Git命令执行失败: {str(e)}")
        return ""
    except Exception as e:
        app.logger.error(f"获取差异失败: {str(e)}")
        return ""

def validate_commit_message(message: str) -> str:
    """验证提交信息格式"""
    if not message:
        return "docs: 自动生成提交信息"

    # 格式规范化处理
    message = message.strip()
    if not re.match(r'^\w+?:', message):
        message = f"fix: {message}"

    # 长度截断
    return message[:72]  # 符合git提交规范的最大长度

# 生成一个提交信息
# 输出：即将提交的信息
@app.route("/commit_create")
def commit_create():
    try:
        project_root = get_project_root()

        # 第一步：确保暂存区干净
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            check=True,
            cwd=str(project_root),
            timeout=20,
            encoding='utf-8',
            errors='replace'
        )

        # 第二步：添加所有变更
        subprocess.run(
            ["git", "add", "--all"],
            check=True,
            cwd=str(project_root),
            timeout=60,
            encoding='utf-8',
            errors='replace'
        )

        # 第三步：获取HEAD的差异内容
        diff_process = subprocess.run(
            ["git", "diff", "--staged", "--no-color"],
            cwd=str(project_root),
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        diff_content = diff_process.stdout

        app.logger.info(f"[DEBUG] 差异内容长度: {len(diff_content)}")

        # 检测空差异内容
        if len(diff_content.strip()) == 0:
            return jsonify({"error": "没有检测到可提交的更改"}), 400

        # 第四步：AI生成提交信息
        try:
            # 简化提示模板
            prompt_template = """
                你是一个专业的软件工程师，正在帮助生成 Git 提交信息。请根据以下代码变更生成简洁明了的提交信息。
                
                要求：
                1. 严格遵守 Conventional Commits 规范（格式：<type>(<scope>): <description>）
                2. 使用恰当的提交类型（feat、fix、docs、style、refactor、test、chore 等）
                3. 描述部分不超过 20 个单词，清晰描述变更目的
                4. 不包含任何解释性文字，只输出最终的提交信息
                
                代码变更：
                {diff_content}
                
                示例：
                fix(auth): resolve password validation issue
                feat(api): add pagination to user endpoint
                refactor(utils): optimize data processing logic
                
                请生成提交信息：
            """

            # 构造AI生成链
            chain = (
                    {"diff_content": RunnablePassthrough()}
                    | PromptTemplate.from_template(prompt_template)
                    | get_llm()
                    | StrOutputParser()
            )

            # 调用AI生成结果
            result = chain.invoke({"diff_content": diff_content})

        except Exception as e:
            app.logger.error(f"AI生成失败: {str(e)}", exc_info=True)
            result = None

        # 第五步：清理结果
        final_result = "更新代码"  # 默认提交信息

        if result:
            # 简单清理 - 移除前后空格和多余空行
            cleaned = result.strip()

            # 只取第一行内容
            if '\n' in cleaned:
                cleaned = cleaned.split('\n', 1)[0].strip()

            # 防止空字符串
            if cleaned:
                final_result = cleaned
            else:
                app.logger.warning("AI生成了空内容，使用默认提交信息")
        else:
            app.logger.warning("AI生成失败，使用默认提交信息")

        return jsonify({
            "status": "success",
            "message": final_result
        })

    except subprocess.TimeoutExpired as e:
        return jsonify({
            "status": "error",
            "message": "操作超时，请重试"
        }), 504
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": f"Git命令失败: {e.stderr if e.stderr else e.stdout}"
        }), 500
    except Exception as e:
        app.logger.error(f"系统级错误: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "系统内部错误"
        }), 500

# 提交所有变更代码
@app.route("/git_commit", methods=["POST"])
def git_commit():
    data = request.json
    msg = data.get("message")
    project_root = get_project_root()

    try:
        # 方案1：最佳方案 - 使用 git add -A 添加所有变更
        subprocess.run(
            ["git", "add", "--all"],  # 替代原来的 -u
            cwd=str(project_root),
            check=True
        )

        # 方案2：仅限当前目录的替代方案
        # subprocess.run(
        #   ["git", "add", "."],
        #   cwd=str(project_root),
        #   check=True
        # )

        # 检查是否有需要提交的变更
        status_check = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True
        )

        if not status_check.stdout.strip():
            return jsonify({
                "status": "info",
                "message": "没有可提交的更改"
            }), 200

        # 提交变更
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(project_root),
            check=True
        )

        # 推送到远程仓库
        push_result = subprocess.run(
            ["git", "push", "origin", "master"],  # 考虑获取当前分支
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True
        )

        return jsonify({
            "status": "success",
            "message": "提交并推送成功",
            "push_output": push_result.stdout
        })

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "未知错误"
        return jsonify({
            "status": "error",
            "message": f"操作失败: {error_msg}"
        }), 500

class ChatHistory(db.Model):
    __tablename__ = 'conversations'
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text)
    roler = db.Column(db.String(10))  # user/bot
    timestamp = db.Column(db.DateTime, default=datetime.now)

# 获取历史聊天记录
@app.route("/get_chat_history")
def get_chat_history():
    try:
        # 获取项目路径
        project_root = get_project_root()

        # 获取原始数据
        raw_data = get_conversation(project_root)

        # 转换为前端需要的格式
        history = [{
            "id": row[0],
            "content": row[1],
            "role": row[2],
            "time": row[3]  # 假设数据库有时间字段
        } for row in raw_data]

        return jsonify({
            "status": "success",
            "history": history
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取失败: {str(e)}"
        }), 500

import git

# 获取一个新的项目
@app.route('/clone_repo', methods=['POST'])
def clone_repo():
    """
    克隆Git仓库到指定路径
    必需参数:
    - repo_url: Git仓库地址 (HTTP/SSH格式)
    - save_path: 项目保存路径 (服务器上的绝对路径)
    """
    data = request.get_json()

    # 验证必需参数
    if not data or 'repo_url' not in data or 'save_path' not in data:
        return jsonify({'error': '缺少必要参数: repo_url 和 save_path'}), 400

    repo_url = data['repo_url']
    save_path = data['save_path']
    branch = data.get('branch')
    username = data.get('username')
    password = data.get('password')

    # 处理认证信息
    if username and password:
        if repo_url.startswith('https://'):
            # 将认证信息嵌入HTTPS URL
            repo_url = repo_url.replace('https://', f'https://{username}:{password}@')
        elif repo_url.startswith('git@'):
            # SSH方式需要提前配置密钥，此处只返回提示
            return jsonify({
                'warning': 'SSH认证请使用密钥方式，请确保服务器已配置SSH密钥',
                'repo_url': repo_url
            }), 200

    try:
        # 清理已存在的目录
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # 执行克隆操作
        clone_args = {}
        if branch:
            clone_args['branch'] = branch

        git.Repo.clone_from(
            url=repo_url,
            to_path=save_path,
        )

        return jsonify({
            'message': '仓库克隆成功',
            'path': save_path,
            'repo': repo_url,
            'branch': branch or '默认分支'
        }), 200

    except git.exc.GitCommandError as e:
        return jsonify({'error': f'Git操作失败: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


def run_flask(port=5000):
    app.run(host="127.0.0.1", port=port, use_reloader=False, threaded=True)

# 初始化embedding模型
    # 初始化模型
print("初始化embedding模型")
embeddings = get_embeddings()

if __name__ == "__main__":
    run_flask()
