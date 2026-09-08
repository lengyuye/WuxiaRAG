import os
import logging
from typing import List

from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag import PROMPT_ANSWER_TEMPLATE

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
	"""生成集成模块 - 负责LLM集成和回答生成"""

	def __init__(self, model_name: str = "deepseek-v4-flash", temperature: float = 0.1, max_tokens: int = 2048):
		"""
		初始化生成集成模块

		Args:
			model_name: 模型名称
			temperature: 生成温度
			max_tokens: 最大token数
		"""
		self.model_name = model_name
		self.temperature = temperature
		self.max_tokens = max_tokens
		self.llm = None
		self.setup_llm()
		self.prompt_answer = ChatPromptTemplate.from_template(PROMPT_ANSWER_TEMPLATE)

	def setup_llm(self):
		"""初始化大语言模型"""
		logger.info(f"正在初始化LLM: {self.model_name}")

		api_key = os.getenv("LLM_API_KEY")
		if not api_key:
			raise ValueError("请设置 LLM_API_KEY 环境变量")

		self.llm = ChatDeepSeek(
			api_key=api_key,
			model=self.model_name,
			temperature=self.temperature,
			max_tokens=self.max_tokens,
			timeout=60,
			max_retries=3,
			reasoning_effort="none",  # 关闭思考模式
			extra_body={"thinking": {"type": "disabled"}}  # 关闭思考模式 可同时指定，更明确
		)

		logger.info("LLM初始化完成")

	def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
		"""
		生成基础回答

		Args:
			query: 用户查询
			context_docs: 上下文文档列表

		Returns:
			生成的回答
		"""
		context = self._build_context(context_docs)
		# 使用LCEL构建链
		chain = (
				{"question": RunnablePassthrough(), "context": lambda _: context}
				| self.prompt_answer
				| self.llm
		)

		response = chain.invoke(query)
		return response.content

	def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
		"""
		生成基础回答 - 流式输出

		Args:
		    query: 用户查询
		    context_docs: 上下文文档列表

		Yields:
		    生成的回答片段
		"""
		context = self._build_context(context_docs)
		chain = (
				{"question": RunnablePassthrough(), "context": lambda _: context}
				| self.prompt_answer
				| self.llm
				| StrOutputParser()
		)

		for chunk in chain.stream(query):
			yield chunk

	def _build_context(self, docs: List[Document], max_length: int = 10000) -> str:
		"""
		构建上下文字符串

		Args:
		    docs: 文档列表
		    max_length: 最大长度

		Returns:
		    格式化的上下文字符串
		"""
		if not docs:
			return "暂无相关小说信息"

		context_parts = []
		current_length = 0


		for i, doc in enumerate(docs, 1):
			# 添加元数据信息
			metadata_info = f"小说信息 {doc.metadata.get('chapter_title','未知章节')}\n"

			# 构建文档文本
			doc_text = f"{metadata_info}\n{doc.page_content}\n"

			# 检查长度限制
			if current_length + len(doc_text) > max_length:
				break

			context_parts.append(doc_text)
			current_length += len(doc_text)

		divider = "\n" + "=" * 50 + "\n"
		return divider + divider.join(context_parts)