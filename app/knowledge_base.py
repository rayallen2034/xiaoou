"""
知识库管理模块
负责文档加载、向量存储和检索工具创建
使用单例模式避免重复初始化
"""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


# 全局缓存变量
_knowledge_db_cache = None
_retriever_tools_cache = None


def init_knowledge_base(force_reload=False):
    """
    初始化知识库 - 加载文档并创建向量存储
    使用缓存机制避免重复加载
    
    Args:
        force_reload: 是否强制重新加载知识库
        
    Returns:
        FAISS: 向量数据库实例，如果知识库为空则返回None
    """
    global _knowledge_db_cache
    
    # 如果已有缓存且不强制重新加载，直接返回
    if _knowledge_db_cache is not None and not force_reload:
        print(f"✅ 使用缓存的知识库，包含 {_knowledge_db_cache.index.ntotal} 个文档片段")
        return _knowledge_db_cache
    
    # 知识库目录（请在此目录放置您的文档）
    knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # 检查是否已有向量数据库
    vector_db_path = os.path.join(knowledge_dir, "faiss_index")
    
    if os.path.exists(vector_db_path):
        # 加载已存在的向量数据库
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        load_dotenv(dotenv_path=env_path)
        
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        _knowledge_db_cache = db  # 缓存知识库
        print(f"✅ 已加载现有知识库，包含 {db.index.ntotal} 个文档片段")
        return db
    
    # 如果没有向量数据库，尝试加载文档
    documents = []
    
    # 支持的文档类型
    loaders = [
        DirectoryLoader(knowledge_dir, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(knowledge_dir, glob="*.md", loader_cls=TextLoader),
        DirectoryLoader(knowledge_dir, glob="*.pdf", loader_cls=PyPDFLoader),
    ]
    
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
            print(f"📄 加载了 {len(docs)} 个 {loader.glob} 文件")
        except Exception as e:
            print(f"⚠️ 加载 {loader.glob} 文件失败: {e}")
    
    if not documents:
        print("ℹ️ 知识库目录为空，请将文档放入 knowledge_base 目录")
        return None
    
    # 分割文档为小块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✂️ 将文档分割为 {len(split_docs)} 个片段")
    
    # 创建嵌入并存储到向量数据库
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(dotenv_path=env_path)
    
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )
    
    db = FAISS.from_documents(split_docs, embeddings)
    
    # 保存向量数据库
    db.save_local(vector_db_path)
    print(f"💾 知识库已保存到 {vector_db_path}")
    
    _knowledge_db_cache = db  # 缓存知识库
    return db


def create_retriever_tools(knowledge_db, force_recreate=False):
    """
    创建知识库检索工具
    使用缓存机制避免重复创建
    
    Args:
        knowledge_db: FAISS 向量数据库实例
        force_recreate: 是否强制重新创建检索工具
        
    Returns:
        list: 检索工具列表
    """
    global _retriever_tools_cache
    
    # 如果已有缓存且不强制重新创建，直接返回
    if _retriever_tools_cache is not None and not force_recreate:
        print("🔧 使用缓存的检索工具")
        return _retriever_tools_cache
    
    retriever_tools = []
    
    if knowledge_db:
        retriever = knowledge_db.as_retriever(search_kwargs={"k": 3})
        retriever_tool = create_retriever_tool(
            retriever,
            name="knowledge_base_search",
            description="""
            搜索知识库获取相关信息。
            当用户询问关于Linux系统管理、CentOS运维、故障排查、性能调优、安全加固等相关问题时，
            请优先使用此工具检索知识库内容。
            """
        )
        retriever_tools.append(retriever_tool)
        _retriever_tools_cache = retriever_tools  # 缓存检索工具
        print("🔧 已创建知识库检索工具")
    
    return retriever_tools


def setup_knowledge_base(force_reload=False, force_recreate=False):
    """
    设置知识库的便捷函数
    使用缓存机制，避免重复初始化
    
    Args:
        force_reload: 是否强制重新加载知识库
        force_recreate: 是否强制重新创建检索工具
        
    Returns:
        tuple: (knowledge_db, retriever_tools)
    """
    knowledge_db = init_knowledge_base(force_reload=force_reload)
    retriever_tools = create_retriever_tools(knowledge_db, force_recreate=force_recreate)
    return knowledge_db, retriever_tools


def clear_cache():
    """清除知识库缓存，强制下次重新初始化"""
    global _knowledge_db_cache, _retriever_tools_cache
    _knowledge_db_cache = None
    _retriever_tools_cache = None
    print("🗑️ 已清除知识库缓存")


def get_knowledge_base_status():
    """获取知识库状态信息"""
    status = {
        "knowledge_db_loaded": _knowledge_db_cache is not None,
        "retriever_tools_created": _retriever_tools_cache is not None,
    }
    
    if _knowledge_db_cache:
        status["document_count"] = _knowledge_db_cache.index.ntotal
    
    if _retriever_tools_cache:
        status["tool_count"] = len(_retriever_tools_cache)
        status["tool_names"] = [tool.name for tool in _retriever_tools_cache]
    
    return status