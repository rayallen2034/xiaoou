from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

# 导入知识库管理模块
from knowledge_base import setup_knowledge_base, get_knowledge_base_status


# 加载环境变量 - 使用绝对路径确保正确加载
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# 多模态模型
multimodel_model = init_chat_model(
    model="qwen3.6-plus",
    model_provider="openai",
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# web搜索工具
web_search = TavilySearch(
    max_results=5,
    topic="general"
)

# ==================== 知识库配置 ====================
# 初始化知识库并创建检索工具（使用缓存机制，避免重复初始化）
knowledge_db, retriever_tools = setup_knowledge_base()

# # 初始化checkpointer（可选）
# db_path = os.path.join(os.path.dirname(__file__), "db", "agent_memory.db")
# os.makedirs(os.path.dirname(db_path), exist_ok=True)
# connection = sqlite3.connect(db_path, check_same_thread=False)
# checkpointer = SqliteSaver(connection)
# checkpointer.setup()


# ==================== 系统提示词 ====================
system_prompt = """
## 角色定位

你是一位资深的 **Linux 操作系统维护专家**，专注于 **CentOS 发行版** 的系统管理、故障排查、性能调优及安全加固领域。

## 工具使用策略（优先级从高到低）

### 1. 知识库检索 (knowledge_base_search)
- **优先使用**此工具检索内部知识库
- 适用于：Linux系统管理、CentOS配置、故障排查、性能调优、安全加固等专业问题
- 知识库包含：操作指南、配置手册、常见问题解答、最佳实践文档

### 2. 网络搜索 (tavily_search)
- 当知识库无法找到答案时使用
- 适用于：最新技术动态、外部资源链接、实时信息查询

### 3. 专业知识
- 仅在以上工具都无法获取信息时使用
- 基于自身专业知识储备提供解答

## 响应规范

### 结构化输出
- 使用 Markdown 格式组织内容
- 代码块使用 ```bash 标记
- 关键步骤用列表清晰呈现

### 安全警示
- 高风险操作前添加 ⚠️ 风险提示
- 建议用户操作前备份数据

### 诚实原则
- 明确说明信息来源
- 无法回答时如实告知，不编造信息

## 专业领域
- 系统管理：用户管理、文件权限、服务配置、计划任务
- 故障排查：日志分析、错误诊断、服务恢复
- 性能调优：资源监控、性能分析、优化策略
- 安全加固：防火墙配置、访问控制、漏洞防护
"""

# 创建智能体（整合所有工具）
all_tools = retriever_tools + [web_search]

agent = create_agent(
    model=multimodel_model,
    tools=all_tools,
    # checkpointer=checkpointer,
    system_prompt=system_prompt
)

print("🚀 Linux运维专家智能体初始化完成")

# 显示知识库状态
status = get_knowledge_base_status()
print(f"📚 知识库状态: {'已加载' if status['knowledge_db_loaded'] else '未配置'}")
if status['knowledge_db_loaded']:
    print(f"📄 文档片段数: {status.get('document_count', 0)}")
print(f"🛠️ 可用工具: {[tool.name for tool in all_tools]}")