import logging
import hashlib
from typing import List, Dict, Any
import jieba
import jieba.posseg as pseg
import json
from collections import defaultdict,Counter

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import DEFAULT_CONFIG
from rag import SearchType, DocumentConfig, PROMPT_DECOMPOSE_TEMPLATE, PROMPT_RERANK_TEMPLATE

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
	"""检索优化模块 - 负责混合检索和过滤"""

	def __init__(self, vectorstore: FAISS, chunks: List[Document]):
		"""
		初始化检索优化模块

		Args:
			vectorstore: FAISS向量存储
			chunks: 文档块列表
		"""
		self.vectorstore = vectorstore
		self.chunks = chunks
		self.document_configs = DocumentConfig.load_all()
		self.setup_retrievers()

	def setup_retrievers(self):
		"""设置向量检索器和BM25检索器"""
		logger.info("正在设置检索器...")

		# 向量检索器
		k = DEFAULT_CONFIG.k
		self.vector_retriever = self.vectorstore.as_retriever(
			search_type="similarity",
			search_kwargs={"k": k}
		)

		# BM25检索器
		self.bm25_retriever = BM25Retriever.from_documents(
			self.chunks,
			k= k,
			preprocess_func= self._preprocess_chinese
		)

		logger.info("检索器设置完成")

	def vector_search(self,query:str, top_k: int = 5):
		"""
		纯向量检索
		:param query:
		:param top_k:
		:return:
		"""
		docs = self.vector_retriever.invoke(query)
		return docs[:top_k]

	def multi_hybrid_search(self,llm:ChatDeepSeek, query: str, top_k: int = 5):
		# 生成多条检索路径
		plan = self._query_decompose(llm,query)
		routes =self.build_retrieval_queries(plan)

		result_lists, weights = [], []
		for r in routes:
			if r["method"] == "vector":
				docs = self.vector_retriever.invoke(r["query"])
			else:
				docs = self.bm25_retriever.invoke(r["query"])
			result_lists.append(docs)
			weights.append(r["weight"])

		merged = self.weighted_rrf(
			result_lists,
			weights=weights,
			top_k=20,
			k=20,
			force_each_top1=True,  # 每路 top1 保底
		)

		final = self.llm_rerank(llm, query, merged, top_k=top_k)
		return final

	@staticmethod
	def _rrf_rerank_new(*result_lists, top_k=5, k=60):
		"""
		使用RRF (Reciprocal Rank Fusion) 算法重排文档
		result_lists: 任意多路检索结果，每路是一个 list[Document]
		top_k: 返回前多少条
		k: RRF 平滑常数
		"""
		scores = defaultdict(float)
		doc_map = {}

		for results in result_lists:
			for rank, doc in enumerate(results, start=1):
				doc_id = RetrievalOptimizationModule.get_doc_id(doc)
				scores[doc_id] += 1.0 / (k + rank)
				doc_map[doc_id] = doc

		sorted_ids = sorted(scores, key=lambda x: -scores[x])
		return [doc_map[i] for i in sorted_ids[:top_k]]

	@staticmethod
	def get_doc_id(doc:Document) -> str:
		return doc.metadata.get("chunk_id")

	def weighted_rrf(self,
			result_lists: List[List],
			weights: List[float] = None,
			top_k: int = 20,
			k: int = 20,
			force_each_top1: bool = False,
	):
		"""
		result_lists: 多路检索结果
		weights: 每路权重，与 result_lists 对齐
		k: RRF 平滑常数，20 比 60 更适合少量路
		force_each_top1: 是否把每路 top1 强制保底进候选
		"""
		if weights is None:
			weights = [1.0] * len(result_lists)
		assert len(weights) == len(result_lists)

		scores = defaultdict(float)
		doc_map = {}

		for w, results in zip(weights, result_lists):
			for rank, doc in enumerate(results, start=1):
				did = RetrievalOptimizationModule.get_doc_id(doc)
				scores[did] += w * 1.0 / (k + rank)
				doc_map[did] = doc

		sorted_ids = sorted(scores, key=lambda x: -scores[x])
		merged = [doc_map[i] for i in sorted_ids]

		if force_each_top1:
			forced, seen = [], set()
			for results in result_lists:
				if results:
					did = RetrievalOptimizationModule.get_doc_id(results[0])
					if did not in seen:
						forced.append(results[0])
						seen.add(did)
			rest = [d for d in merged if RetrievalOptimizationModule.get_doc_id(d) not in seen]
			merged = forced + rest

		return merged[:top_k]

	def _query_decompose(self,llm:ChatDeepSeek, query: str) -> Dict[str, Any]:
		"""
		查询分解
		:param llm:
		:param query:
		:return:
		"""
		decompose_prompt = ChatPromptTemplate.from_template(PROMPT_DECOMPOSE_TEMPLATE)

		chain = (
				{"question": RunnablePassthrough()}
				| decompose_prompt
				| llm
				| StrOutputParser()
		)

		response = chain.invoke(query)
		ok, resp, err = self._check_json(response)
		if not ok:
			# 兜底：只用原问题
			print("查询分解失败，使用原始问题")
			return {
				"strip_query": query,
				"hyde": "",
				"scene_synonyms": [],
				"cooccur_entities": [],
			}
		print(f"查询分解成功：{resp}")
		return resp

	def build_retrieval_queries(self,plan: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""
		返回 4 路检索任务：每路有自己的 query、检索方式、权重
		"""
		routes = []

		# 路1：去噪原问 —— 语义向量，权重中等
		if plan.get("strip_query"):
			routes.append({
				"name": "strip_query",
				"query": plan["strip_query"],
				"method": "vector",
				"weight": 1.0,
			})

		# 路2：HyDE —— 语义向量，权重最高（最贴近原文）
		if plan.get("hyde"):
			routes.append({
				"name": "hyde",
				"query": plan["hyde"],
				"method": "vector",
				"weight": 2.5,
			})

		# 路3：场景同义词 —— 关键词/BM25，权重高（精确匹配原文词）
		if plan.get("scene_synonyms"):
			routes.append({
				"name": "scene_synonyms",
				"query": " ".join(plan["scene_synonyms"]),
				"method": "bm25",
				"weight": 2.0,
			})

		# 路4：共现实体扩展 —— 向量 + BM25 都跑，取并集，权重中等
		if plan.get("cooccur_entities"):
			ent_query = " ".join(plan["cooccur_entities"])
			routes.append({
				"name": "cooccur_vector",
				"query": ent_query,
				"method": "vector",
				"weight": 1.0,
			})
			routes.append({
				"name": "cooccur_bm25",
				"query": ent_query,
				"method": "bm25",
				"weight": 1.0,
			})

		return routes

	RERANK_PROMPT = """你是武侠小说 RAG 的重排器。下面是一个用户问题和若干候选原文块。
	请按"能否回答用户问题"排序，输出 JSON：{{"order": [最相关的候选编号, ...]}}

	用户问题：{question}

	候选：
	{candidates}

	只输出 JSON，不要解释。
	"""

	def llm_rerank(self,llm:ChatDeepSeek, question: str, docs: List, top_k: int = 5) -> List:
		"""
		大模型重排答案
		:param question:
		:param docs:
		:param top_k:
		:return:
		"""
		if len(docs) <= top_k:
			return docs

		cand_text = "\n\n".join(
			f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs)
		)
		prompt = ChatPromptTemplate.from_template(PROMPT_RERANK_TEMPLATE)
		chain = prompt | llm | StrOutputParser()

		try:
			raw = chain.invoke({"question": question, "candidates": cand_text})
			raw = raw.strip().strip("```").lstrip("json").strip()
			order = json.loads(raw)["order"]
			reranked = [docs[i] for i in order if 0 <= i < len(docs)]
			# 补齐遗漏的
			seen = {id(d) for d in reranked}
			reranked += [d for d in docs if id(d) not in seen]
			return reranked[:top_k]
		except Exception as e:
			print(f"rerank 失败：{e}")
			return docs[:top_k]

	@staticmethod
	def _check_json(s):
		try:
			data = json.loads(s)
			return True, data, None
		except json.JSONDecodeError as e:
			return False, None, f"JSON 解析错误: {e}"
		except TypeError as e:
			return False, None, f"类型错误: {e}"

	@staticmethod
	def _preprocess_chinese(text):
		"""
		专用于给中文分词的方法
		:return:
		"""
		tokens = list(jieba.cut(text, cut_all=False))  # 这里用精确模式比较好
		# print(f"Preprocessing: '{text}' -> Tokens: {tokens}") 测试使用
		return tokens