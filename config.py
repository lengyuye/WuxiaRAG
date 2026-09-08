"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # C:/GitProject/WuxiaRAG

@dataclass
class RAGConfig:

	# 系统名字
	sys_name = "武侠小说RAG系统"

	# 路径配置
	data_path: str = str(BASE_DIR / "data" / "documents")
	index_save_path: str = str (BASE_DIR/"vector_index")

	#知识库源文件类型(目前仅支持一种文件)
	file_type: str = "pdf"

	# 模型配置
	embedding_model: str = "BAAI/bge-small-zh-v1.5"
	llm_model: str = "deepseek-flash"

	# 检索配置
	k:int = 5  	#搜索范围（向量+BM25混合检索，所以实际检索范围是k*2） 默认是5
	top_k: int = 5 #返回的结果数量，默认值是3，

	# 生成配置
	temperature: float = 1.0
	max_tokens: int = 8192

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()