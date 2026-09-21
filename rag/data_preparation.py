"""
数据准备模块
"""

import logging
import uuid
from typing import List, Dict, Any
from langchain_pdfmux import PDFMuxLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import PureWindowsPath
import glob
import re

logger = logging.getLogger(__name__)

class DataPreparationModule:
	def __init__(self, data_path: str,file_type: str):
		"""
		初始化数据准备模块

		Args:
			data_path: 数据文件夹路径
		"""
		self.data_path = data_path
		self.file_type = file_type
		self.documents: List[Document] = []  # 父文档（按章节拆分的小说）
		self.chunks: List[Document] = []  # 子文档（章节内分割的小块）
		self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射

	def _split_chapters(self,book_path: str, text: str) -> list[Document]:
		"""
		把文档按章节拆分
		:param text:
		:return:
		"""
		book_filename = PureWindowsPath(book_path).name
		book_title = book_filename.replace(".pdf", "")

		# 在“第XX章 标题”前切分，保留标题
		pattern = re.compile(r"(?=第\d+章\s+[^\n]*\n)")
		parts = pattern.split(text)

		parents = []
		for part in parts:
			part = part.strip()
			if not part:
				continue

			# 提取章节号和标题
			m = re.match(r"第(\d+)章\s+([^\n]+)", part)
			if m:
				chapter_num = m.group(1)
				chapter_title = f"第{chapter_num}章 {m.group(2).strip()}"
			else:
				chapter_num = "00"
				chapter_title = "前言"

			chapter_id = f"{book_title}-ch{chapter_num}"
			metadata = {
				#"book": book_title,
				#"chapter_num": chapter_num,
				"chapter_title": chapter_title,
				"chapter_id": chapter_id,
				"source": f"{book_title}_{chapter_title}",
			}

			parents.append(Document(page_content=part, metadata=metadata))

		return parents

	def load_documents(self) -> List[Document]:
		"""
		       加载文档数据

		       Returns:
		           加载的文档列表
		       """
		logger.info(f"正在从 {self.data_path} 加载文档...")

		pdf_files = glob.glob(f"{self.data_path}/*.{self.file_type}")

		documents = []

		for pdf_path in pdf_files:
			try:
				loader = PDFMuxLoader(pdf_path, quality="standard")
				docs = loader.load()
				if len(docs) < 1:
					logger.error(f"{pdf_path} 没有获取到文档")
					continue
				# 这里默认一个PDF只会读取到一个Document
				full_content = docs[0].page_content
				# 按章节拆分Document
				documents_chapter = self._split_chapters(pdf_path, full_content)

				#大块分割，作为真正的父文档
				big_chunk_splitter = self._get_child_spliter(
					chunk_size=2000,
					chunk_overlap=320,  # 重叠 10%~20%
				)
				for document in documents_chapter:
					big_chunks = big_chunk_splitter.split_documents([document])
					chapter_id = document.metadata["chapter_id"]
					source = document.metadata["source"]
					chapter_title = document.metadata["chapter_title"]
					# 设置元数据
					for i, chunk in enumerate(big_chunks):
						chunk.metadata.update({
							"source": source,
							"chapter_title":chapter_title,
							"parent_id": f"{chapter_id}-{i}",
							"doc_type": "parent",  # 标记为父文档
						})
					documents.extend(big_chunks)
				logger.debug(f"已处理: {pdf_path}，共 {len(documents_chapter)} 章")
			except Exception as e:
				logger.warning(f"读取文件 {pdf_path} 失败: {e}")

		self.documents = documents
		logger.info(f"成功加载 {len(documents)} 个文档")
		return documents

	def _get_child_spliter(self,chunk_size:int,chunk_overlap:int)->RecursiveCharacterTextSplitter:
		"""
		获取文本分割器
		:param chunk_size:块大小
		:param chunk_overlap:块重叠
		:return:
		"""
		# 按中文的习惯进行分割
		split_on = [
			"\n\n",  # 段落
			"\n",  # 换行
			"。",  # 句号
			"！",  # 感叹号
			"？",  # 问号
			"；",  # 分号
			"，",  # 逗号
			" ",  # 空格
			"",  # 字符兜底
		]

		child_splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,  # 中文建议 300~800
			chunk_overlap=chunk_overlap,  # 重叠 10%~20%
			length_function=len,
			separators=split_on,
			is_separator_regex=False,
		)
		return child_splitter

	def chunk_documents(self) -> List[Document]:
		"""
		分块
		:param parents: 待分块的父文档列表
		:return:
		"""
		child_splitter = self._get_child_spliter(
			chunk_size=500,  # 中文建议 300~800
			chunk_overlap=80,  # 重叠 10%~20%
		)

		all_chunks = []

		for parent in self.documents:
			# 为每个子块建立与父文档的关系
			parent_id = parent.metadata["parent_id"]
			try:
				chunks = child_splitter.split_documents([parent])
				for i, chunk in enumerate(chunks):
					# 为子块分配唯一ID
					child_id = str(uuid.uuid4())

					# 合并父文档元数据和新的元数据
					chunk.metadata.update(parent.metadata)
					chunk.metadata.update({
						"chunk_id": child_id,
						"parent_id": parent_id,
						"doc_type": "child",  # 标记为子文档
						"chunk_index": i,  # 在父文档中的位置
						"chunk_size":len(chunk.page_content),
					})
					# 建立父子映射关系
					self.parent_child_map[child_id] = parent_id
					all_chunks.append(chunk)
			except Exception as e:
				logger.warning(f"文档 {parent_id} 分割失败: {e}")
				# 如果分割失败，将整个文档作为一个chunk
				all_chunks.append(parent)

		self.chunks = all_chunks
		logger.info(f"分割完成，生成 {len(all_chunks)} 个结构化块")
		return all_chunks

	def get_statistics(self) -> Dict[str, Any]:
		"""
		获取数据统计信息

		Returns:
			统计信息字典
		"""
		if not self.documents:
			return {}
		return {
			'total_documents': len(self.documents),
			'total_chunks': len(self.chunks),
			'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(
				self.chunks) if self.chunks else 0
		}

	def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
		"""
		根据子块获取对应的父文档（智能去重）

		Args:
		    child_chunks: 检索到的子块列表

		Returns:
		    对应的父文档列表（去重，按相关性排序）
		"""
		# 统计每个父文档被匹配的次数（相关性指标）
		parent_relevance = {}
		parent_docs_map = {}

		# 收集所有相关的父文档ID和相关性分数
		for chunk in child_chunks:
			parent_id = chunk.metadata.get("parent_id")
			if parent_id:
				# 增加相关性计数
				parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

				# 缓存父文档（避免重复查找）
				if parent_id not in parent_docs_map:
					for doc in self.documents:
						if doc.metadata.get("parent_id") == parent_id:
							parent_docs_map[parent_id] = doc
							break

		# 按相关性排序（匹配次数多的排在前面）
		sorted_parent_ids = sorted(
			parent_relevance.keys(),
			key=lambda x: parent_relevance[x],
			reverse=True
		)

		# 构建去重后的父文档列表
		parent_docs = []
		for parent_id in sorted_parent_ids:
			if parent_id in parent_docs_map:
				parent_docs.append(parent_docs_map[parent_id])

		# 收集父文档名称和相关性信息用于日志
		parent_info = []
		for doc in parent_docs:
			parent_id = doc.metadata.get('parent_id')
			relevance_count = parent_relevance.get(parent_id, 0)
			parent_info.append(f"{parent_id}({relevance_count}块)")

		logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
		return parent_docs

	def get_chunks_by_content(self,content:str)->List[Document]:
		"""
		根据内容 返回子块
		:param content:
		:return:
		"""
		chunks = []
		for chunk in self.chunks:
			if content in chunk.page_content:
				chunks.append(chunk)
		return chunks

