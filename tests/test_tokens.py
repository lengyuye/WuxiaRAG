from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import jieba

def test_split_default():
	"""
	测试默认分词
	:return:
	"""

	# 假设你有一个 retriever 实例
	docs = [Document(page_content="LangChain是一个用于构建大语言模型应用的框架。")]
	retriever = BM25Retriever.from_documents(docs)

	# 直接调用 retriever 的预处理函数
	tokens = retriever.preprocess_func("一千个头 神仙姊姊")
	print(tokens)
	# 输出 ['一千个头', '神仙姊姊']

	tokens = retriever.preprocess_func("段誉在无量山洞里磕了一千个头，是向谁磕的？")
	print(tokens)
	# 输出 ['段誉在无量山洞里磕了一千个头，是向谁磕的？']


def debug_preprocess(text):
    tokens = list(jieba.cut(text,cut_all=False)) # 这里用精确模式比较好
    print(f"Preprocessing: '{text}' -> Tokens: {tokens}")
    return tokens

def test_split_custom():
	"""
	测试自定义分词
	:return:
	"""
	docs = [Document(page_content="LangChain是一个用于构建大语言模型应用的框架。")]
	retriever = BM25Retriever.from_documents(docs, preprocess_func=debug_preprocess)

	# 执行查询时，会打印出查询语句的分词结果
	results = retriever.invoke("一千个头 神仙姊姊")
	# 输出 Tokens: ['一千个', '头', ' ', '神仙', '姊姊']
	results = retriever.invoke("段誉在无量山洞里磕了一千个头，是向谁磕的？")
	# 输出  Tokens: ['段誉', '在', '无量山', '洞里', '磕', '了', '一千个', '头', '，', '是', '向', '谁', '磕', '的', '？']

test_split_custom()