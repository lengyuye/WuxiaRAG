from enum import Enum, auto, unique

class SearchType(Enum):
	"""RAG的检索类型"""
	VECTOR = 1
	BM25 = 2
	RRF = 3 #混合 前两种，再加上RRF倒数