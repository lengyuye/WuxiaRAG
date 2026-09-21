import os
from pathlib import Path
import traceback
from typing import List

from langchain_core.documents import Document

from config import RAGConfig, DEFAULT_CONFIG
from tools.language_tool import LanguageTool
from . import SearchType
from rag.document_config import DocumentConfig

from .data_preparation import DataPreparationModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule
from .generation_integration import GenerationIntegrationModule
import random


class WuxiaRAGSystem:
	"""武侠RAG系统"""

	def __init__(self, config: RAGConfig = None):
		"""
		初始化RAG系统

		Args:
			config: RAG系统配置，默认使用DEFAULT_CONFIG
		"""
		self.config = config or DEFAULT_CONFIG
		self.data_module = None
		self.index_module = None
		self.retrieval_module = None
		self.generation_module = None
		self.init_finished = None

		# 检查数据路径
		if not Path(self.config.data_path).exists():
			raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

		# 检查API密钥
		if not os.getenv("LLM_API_KEY"):
			raise ValueError("请设置 LLM_API_KEY 环境变量")

	def initialize_system(self):
		"""初始化所有模块"""
		print("🚀 正在初始化RAG系统...")

		# 1. 初始化数据准备模块
		print("初始化数据准备模块...")
		self.data_module = DataPreparationModule(self.config.data_path, self.config.file_type)

		# 2. 初始化索引构建模块
		print("初始化索引构建模块...")
		self.index_module = IndexConstructionModule(
			model_name=self.config.embedding_model,
			index_save_path=self.config.index_save_path
		)

		# 3. 初始化生成集成模块
		print("🤖 初始化生成集成模块...")
		self.generation_module = GenerationIntegrationModule(
			model_name=self.config.llm_model,
			temperature=self.config.temperature,
			max_tokens=self.config.max_tokens
		)

	def build_knowledge_base(self):
		"""构建知识库"""
		print("\n正在构建知识库...")

		# 1. 尝试加载已保存的索引
		vectorstore = self.index_module.load_index()

		# 2. 加载文档
		print("加载武侠小说文档...")
		self.data_module.load_documents()

		# 3. 文本分块
		print("进行文本分块...")
		chunks = self.data_module.chunk_documents()

		if vectorstore is not None:
			print("✅ 成功加载已保存的向量索引！")
		else:
			print("未找到已保存的索引，开始构建新索引...")
			# 4. 构建向量索引
			print("构建向量索引...")
			vectorstore = self.index_module.build_vector_index(chunks)

			# 5. 保存索引
			print("保存向量索引...")
			self.index_module.save_index()

		# 6. 初始化检索优化模块
		print("初始化检索优化...")
		self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

		# 7. 显示统计信息
		stats = self.data_module.get_statistics()
		print(f"\n📊 知识库统计:")
		print(f"	文档总数: {stats['total_documents']}")
		print(f"	文本块数: {stats['total_chunks']}")
		print(f"	平均块大小: {stats['avg_chunk_size']}")
		print("✅ 知识库构建完成！")

	def ask_question(self, question: str, stream: bool = False, search_type: SearchType = SearchType.RRF):
		"""
		回答用户问题

		Args:
			question: 用户问题
			stream: 是否使用流式输出

		Returns:
			生成的回答或生成器
		"""
		if not all([self.retrieval_module, self.generation_module]):
			raise ValueError("请先构建知识库")

		print(f"\n❓ 用户问题: {question}")

		print("问题预处理")
		document_name, question = self.preprocess_question(question)
		print("🔍 检索相关文档...")

		# 使用查询分解，进行多路查询
		relevant_chunks = self.retrieval_module.multi_hybrid_search(self.generation_module.llm, question, top_k=self.config.top_k)

		# 显示检索到的子块信息
		if relevant_chunks:
			chunk_info = []
			for i, chunk in enumerate(relevant_chunks):
				# 尝试从内容中提取章节标题
				parent_id = chunk.metadata['parent_id']
				chunk_index = chunk.metadata['chunk_index']
				content_preview = chunk.page_content[:10].strip()
				chunk_info.append(f"{parent_id}-{chunk_index}({content_preview}...)")
			print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
		else:
			print(f"没有找到相关文档块")
			return "抱歉，没有找到相关的信息。请尝试其他关键词。"

		print("获取父文档...")
		relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

		# 显示找到的文档名称
		doc_names = []
		for doc in relevant_docs:
			parent_id = doc.metadata.get('parent_id', '未知id')
			doc_names.append(parent_id)
		# 调试使用
		# print(f"找到文档{parent_id}：")
		# print(f"内容：{doc.page_content}")

		if doc_names:
			# print(f"找到文档: {', '.join(doc_names)}")
			pass
		else:
			print(f"对应 {len(relevant_docs)} 个完整文档")

		print("✍️ 生成详细回答...")

		if stream:
			return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
		else:
			return self.generation_module.generate_basic_answer(question, relevant_docs)

	def preprocess_question(self, question: str) -> tuple[str, str]:
		document_name = "天龙八部"  # 目前小说名字先写死
		# 判断小说是否为简体还是繁体，如果是繁体，需要转换问题
		is_simplified = True
		cfg = DocumentConfig.get_by_name(document_name)
		if cfg:
			is_simplified = cfg.is_simplified
		if not is_simplified:
			# 简体转繁体
			question = LanguageTool.simplified_to_traditional(question)
		return document_name, question

	def run_interactive(self):
		"""运行交互式问答"""
		print("=" * 60)
		print(f"🍽️  {self.config.sys_name} - 交互式问答  🍽️")
		print("=" * 60)

		# 初始化系统，然后构建知识库
		self.init_systems_and_build_knowledge_base()


		print("\n交互式问答 (输入'退出'结束):")

		while True:
			try:
				user_input = input("\n您的问题: ").strip()
				if user_input.lower() in ['退出', 'quit', 'exit', '']:
					break

				# 询问是否使用流式输出
				stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
				use_stream = stream_choice != 'n'

				print("\n回答:")
				if use_stream:
					# 流式输出
					for chunk in self.ask_question(user_input, stream=True):
						print(chunk, end="", flush=True)
					print("\n")
				else:
					# 普通输出
					answer = self.ask_question(user_input, stream=False)
					print(f"{answer}\n")

			except KeyboardInterrupt:
				break
			except Exception as e:
				print(f"处理问题时出错: {e}")
				traceback.print_exc()

		print(f"\n感谢使用{self.config.sys_name}！")

	def init_systems_and_build_knowledge_base(self):
		"""
		初始化系统,然后构建知识库
		:return:
		"""
		if not self.init_finished:
			# 初始化系统
			self.initialize_system()
			# 构建知识库
			self.build_knowledge_base()
			self.init_finished = True

	def get_chunks_for_eval(self)->List[Document]:
		"""
		 获取文档数据,仅用于评估数据
		Returns:
			获取的文档列表
		 """
		self.data_module = DataPreparationModule(self.config.data_path, self.config.file_type)
		self.data_module.load_documents()
		# 3. 文本分块
		print("进行文本分块...")
		chunks = self.data_module.chunk_documents()


		# 目标取 10 个文档
		random.seed(42)

		target_count = DEFAULT_CONFIG.eval_sample_count
		step = max(1, len(chunks) // target_count)

		sampled_docs = chunks[::step][:target_count]

		# 如果不够 10 个，再随机补几个
		if len(sampled_docs) < target_count:
			remaining = [d for d in chunks if d not in sampled_docs]
			sampled_docs += random.sample(remaining, target_count - len(sampled_docs))

		print(f"采样了 {len(sampled_docs)} 个文档，覆盖章节：")
		for doc in sampled_docs:
			print(doc.metadata.get('source', '未知'))
		return sampled_docs

	def get_single_rag_output_for_eval(self,question):
		"""
		获取RAG输出用于评估
		:param question:
		:return: 回答相关的上下文，回答
		"""

		document_name, question_process = self.preprocess_question(question)

		# 调用检索器,检索子块
		relevant_chunks = self.retrieval_module.multi_hybrid_search(self.generation_module.llm, question_process,top_k=self.config.top_k)
		# 纯向量检索，用于测试
		# relevant_chunks = self.retrieval_module.vector_search(question,top_k=self.config.top_k)

		# 获取父块
		relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

		# 调用生成链，生成回答
		answer_output = self.generation_module.generate_basic_answer(question_process, relevant_docs)
		return relevant_chunks,answer_output

	def get_rag_output_for_eval(self,questions):
		"""
		获取RAG输出用于评估
		:param questions:
		:return:
		"""
		responses = []
		retrieved_contexts = []
		for q in questions:
			print(f"开始回答问题：{q}")
			relevant_chunks,answer_output = self.get_single_rag_output_for_eval(q)
			responses.append(answer_output)
			retrieved_contexts.append([doc.page_content for doc in relevant_chunks])
		return responses, retrieved_contexts

	def test_query(self, query: str,search_type:SearchType = SearchType.RRF):
		"""
		测试方法，用于测试查询功能
		:return:
		"""
		self.init_systems_and_build_knowledge_base()
		# 普通输出
		answer = self.ask_question(query, stream=False, search_type=search_type)
		print(f"{search_type} answer:{answer}\n")

	def test_get_chunks(self,content:str):
		"""
		测试方法，根据内容找对应子块
		:param content:
		:return:
		"""
		self.init_systems_and_build_knowledge_base()
		chunks= self.data_module.get_chunks_by_content(content)
		if not chunks:
			print("没有找到对应的块")
			return
		print(f"搜索 ‘{content}’，找到 {len(chunks)} 个块")
		for chunk in chunks:
			print(f"metadata: {chunk.metadata}")
			print(f"page_content: {chunk.page_content}")
			print()
