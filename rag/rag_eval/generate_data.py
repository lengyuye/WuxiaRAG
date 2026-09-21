"""
RAG 评估测试集生成脚本 --  generate_data.py

使用 RAGAS 框架，基于武侠小说文档自动生成中文的单跳/多跳问答测试集，
用于评估 RAG 系统的检索与生成效果。

运行前提:
    - 本地 Ollama 服务已启动（默认监听 11434 端口），并已拉取 qwen3.5:4b 模型
    - 可通过 ModelScope 下载 embedding 模型 (BAAI/bge-small-zh-v1.5)

输出:
    - datasets/eval_dataset.csv（utf-8-sig 编码，Excel 可直接打开）
"""

import asyncio
import os
from dotenv import load_dotenv

# 关闭 RAGAS 遥测上报（离线/隐私环境），必须在导入 ragas 之前设置
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from ragas.testset.persona import Persona
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
	SingleHopSpecificQuerySynthesizer,
	MultiHopAbstractQuerySynthesizer,
	MultiHopSpecificQuerySynthesizer,
)
from ragas import RunConfig
from modelscope import snapshot_download
from ragas.llms import llm_factory
from ragas.cache import DiskCacheBackend
from openai import AsyncOpenAI

from config import DEFAULT_CONFIG
from rag import WuxiaRAGSystem

# 从 .env 文件加载环境变量（如需 API Key 等）
load_dotenv()

# 磁盘缓存：缓存 LLM 调用结果，避免重复生成（使用新的缓存目录，避免读到旧的英文缓存）
cache = DiskCacheBackend(cache_dir="./cache_zh_v1")

# ---- 1. 加载 RAG 系统的文档分块，作为测试集的数据源 ----
rag_system = WuxiaRAGSystem()
chunks = rag_system.get_chunks_for_eval()
for chunk in chunks:
	# RAGAS 按 filename 字段采样文档，这里把 source 字段复制为 filename
	chunk.metadata['filename'] = chunk.metadata['source']

# ---- 2. 配置 LLM：通过 OpenAI 兼容接口连接本地 Ollama 服务 ----
# Ollama 本地服务不校验 API Key，任意值即可
client = AsyncOpenAI(
	api_key="ollama",
	base_url="http://localhost:11434/v1",
	timeout=300.0,
)

# 用 RAGAS 的工厂方法创建 LLM 封装，并挂上磁盘缓存避免重复调用
llm = llm_factory(
	"qwen3.5:4b",  # Ollama 中的本地模型名，需与实际拉取的模型一致
	client=client,
	max_tokens=8192,
	reasoning_effort="none",  # 小模型关闭思维链，节省生成成本
	cache=cache,
)

# ---- 3. 配置 Embedding 模型（从 ModelScope 下载到本地） ----
model_dir = snapshot_download(DEFAULT_CONFIG.embedding_model)
ragas_embeddings = HuggingFaceEmbeddings(model=model_dir, cache=cache)

# ---- 4. 定义模拟提问的角色（Persona），用于生成多样化的提问视角 ----
persona_list = [
	Persona(name="武侠爱好者", role_description="熟悉金庸古龙作品的武侠迷"),
	Persona(name="文学研究者", role_description="研究武侠文学的专业学者"),
	Persona(name="普通读者", role_description="对武侠世界好奇的入门读者"),
]

# 测试集生成器：组合 LLM + Embedding + 角色列表
generator = TestsetGenerator(
	llm=llm,
	embedding_model=ragas_embeddings,
	persona_list=persona_list,
)

# ---- 5. 关键修改：手动实例化合成器 + 中文适配 ----
# RAGAS 默认合成器使用英文指令模板，这里手动实例化并在下方替换为中文指令
single_hop = SingleHopSpecificQuerySynthesizer(llm=llm)        # 单跳、具体问题
multi_hop_abstract = MultiHopAbstractQuerySynthesizer(llm=llm) # 多跳、抽象问题
multi_hop_specific = MultiHopSpecificQuerySynthesizer(llm=llm) # 多跳、具体问题

# 翻译后的指令（顺序与上方三个合成器一一对应，用于覆盖 query_answer_generation_prompt）
translate_instruction = ['''
基于指定条件（角色、术语、风格、长度）和所提供的上下文，生成一个单跳查询和答案。确保答案完全忠实于上下文，仅使用直接来自所提供上下文的信息。
### 指令：
1.**生成查询**：基于上下文、角色、术语、风格和长度，创建一个与角色视角一致并包含该术语的问题。
2.**生成答案**：仅使用所提供上下文中的内容，针对该查询构建一个详细答案。不要添加任何未包含在上下文中或无法从上下文中推断出的信息。
3.**附加上下文（如提供）**：如果提供了 llm_context，请将其作为生成何种类型问题（例如比较类问题、如何做类问题、应用型问题）以及如何相应组织答案的指导。仍须确保内容仅来自所提供的上下文。
''',
'''基于指定条件（角色、主题、风格、长度）和所提供的上下文，生成一个多跳查询和答案。主题代表一组从上下文中提取或生成的短语，这些短语突出显示了所选上下文对于创建多跳查询的适用性。确保查询明确包含这些主题。
### 指令：
1.**生成多跳查询**：使用所提供的上下文片段和主题，形成一个需要组合多个片段信息（例如 <1-hop> 和 <2-hop>）的查询。确保查询明确包含一个或多个主题，并反映它们与上下文的相关性。
2.**生成答案**：仅使用所提供上下文中的内容，为该查询创建一个详细且忠实的答案。避免添加未直接出现在所给上下文中或无法从中推断出的信息。
3.**多跳上下文标签**：
	-每个上下文片段标记为 <1-hop>、<2-hop> 等。
	-确保查询使用至少两个片段中的信息，并将它们有意义地连接起来。
4.**附加上下文（如提供）**：如果提供了 llm_context，请将其作为生成何种类型问题（例如比较类问题、因果类问题、应用型问题）以及如何相应组织答案的指导。仍须确保内容仅来自所提供的上下文。
						 ''',
'''基于指定条件（角色、主题、风格、长度）和所提供的上下文，生成一个多跳查询和答案。主题代表一组从上下文中提取或生成的短语，这些短语突出显示了所选上下文对于创建多跳查询的适用性。确保查询明确包含这些主题。
### 指令：
1.**生成多跳查询**：使用所提供的上下文片段和主题，形成一个需要组合多个片段信息（例如 <1-hop> 和 <2-hop>）的查询。确保查询明确包含一个或多个主题，并反映它们与上下文的相关性。
2.**生成答案**：仅使用所提供上下文中的内容，为该查询创建一个详细且忠实的答案。避免添加未直接出现在所给上下文中或无法从中推断出的信息。
3.**多跳上下文标签**：
	 -每个上下文片段标记为 <1-hop>、<2-hop> 等。
	 -确保查询使用至少两个片段中的信息，并将它们有意义地连接起来。
4.**附加上下文（如提供）**：如果提供了 llm_context，请将其作为生成何种类型问题（例如比较类问题、因果类问题、应用型问题）以及如何相应组织答案的指导。仍须确保内容仅来自所提供的上下文。
												  ''']


# 6. 定义适配函数（关键步骤）：将三个合成器的 prompt 适配为中文并写回实例
async def adapt_query_synthesizers():
	# 对每个合成器执行：适配语言 -> 替换中文指令 -> 写入合成器
	i = 0
	for synthesizer in [single_hop, multi_hop_abstract, multi_hop_specific]:
		# 第一步：adapt_prompts 返回一个按 "chinese" 适配后的 prompt 字典
		prompts = await synthesizer.adapt_prompts("chinese", llm=llm)

		# 第二步：手动把自动生成指令替换为翻译后的中文指令
		if "query_answer_generation_prompt" in prompts:
			prompt = prompts["query_answer_generation_prompt"]
			prompt.instruction = translate_instruction[i]  # 替换为中文
			# print(f"prompt.instruction:{prompt.instruction}")
			prompts["query_answer_generation_prompt"] = prompt
			i = i + 1

		print(f"prompts:{prompts}" )
		# 第三步：set_prompts 将适配后的 prompt 应用到合成器实例上
		synthesizer.set_prompts(**prompts)


asyncio.run(adapt_query_synthesizers())

# ---- 7. 配置查询类型占比，并生成测试集 ----
query_distribution = [
    (single_hop, 0.4),          # 40% 单跳、具体问题
    (multi_hop_abstract, 0.3),  # 30% 多跳、抽象问题
    (multi_hop_specific, 0.3),  # 30% 多跳、具体问题
]

# 运行配置：单次请求超时 600 秒、1 个并发 worker、最多重试 2 次
run_config = RunConfig(timeout=600, max_workers=1, max_retries=2)

testset = generator.generate_with_chunks(
    chunks,
    testset_size=15,  # 生成的测试样本数量
    query_distribution=query_distribution,
    run_config=run_config,
)

# ---- 8. 转换为 DataFrame 并预览结果 ----
test_df = testset.to_pandas()
print(test_df.columns.tolist())
print(test_df.head())

# 保存为 CSV（utf-8-sig 带 BOM，Excel 打开不乱码；index=False 不写行号）
test_df.to_csv("datasets/eval_dataset.csv", index=False, encoding="utf-8-sig")
