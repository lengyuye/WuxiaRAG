"""
	运行评估 脚本
"""


import asyncio
import ast
import time

from config import DEFAULT_CONFIG

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import Dataset
import pandas as pd
from ragas import experiment
from ragas.llms import llm_factory
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from modelscope import snapshot_download
import traceback


@experiment()
async def wuxia_rag_experiment(row, semaphore: asyncio.Semaphore):
	try:
		t0 = time.time()
		print(f"[{t0:.1f}] enter row, waiting semaphore")

		question = row["user_input"]
		response = row["response"]
		retrieved_contexts = row["retrieved_contexts"]
		reference = row["reference"]

		if isinstance(retrieved_contexts, str):
			retrieved_contexts = ast.literal_eval(retrieved_contexts)

		async with semaphore:
			print(f"[{time.time() - t0:.1f}s] got semaphore")

			t = time.time()
			faithfulness_result = await faithfulness_metric.ascore(
				user_input=question, response=response, retrieved_contexts=retrieved_contexts,
			)
			print(f"[{time.time() - t0:.1f}s] faithfulness done, took {time.time() - t:.1f}s")

			t = time.time()
			answer_relevancy_result = await answer_relevancy_metric.ascore(
				user_input=question, response=response,
			)
			print(f"[{time.time() - t0:.1f}s] answer_relevancy done, took {time.time() - t:.1f}s")

			t = time.time()
			context_precision_result = await context_precision_metric.ascore(
				user_input=question, reference=reference, retrieved_contexts=retrieved_contexts,
			)
			print(f"[{time.time() - t0:.1f}s] context_precision done, took {time.time() - t:.1f}s")

			t = time.time()
			context_recall_result = await context_recall_metric.ascore(
				user_input=question, retrieved_contexts=retrieved_contexts, reference=reference,
			)
			print(f"[{time.time() - t0:.1f}s] context_recall done, took {time.time() - t:.1f}s")

		return {
			**row,
			"faithfulness": faithfulness_result.value,
			"answer_relevancy": answer_relevancy_result.value,
			"context_precision": context_precision_result.value,
			"context_recall": context_recall_result.value,
		}
	except Exception as e:
		print("!!! EXCEPTION IN ROW !!!")
		print("type:", type(e).__name__)
		print("repr:", repr(e))
		print("args:", e.args)
		traceback.print_exc()
		raise


async def main():
	# 1. 创建 LLM 和 embeddings
	client = AsyncOpenAI(
		api_key="ollama",
		base_url="http://localhost:11434/v1",
		timeout=300.0,
	)

	model_name = "qwen2.5:3b"

	llm= llm_factory(model_name, client=client,max_tokens=4096, max_retries=3)

	model_dir = snapshot_download(DEFAULT_CONFIG.embedding_model)
	embeddings = HuggingFaceEmbeddings(model=model_dir)

	# 2. 实例化指标（模块级别或 main 内部，只做一次）
	global faithfulness_metric, answer_relevancy_metric, context_precision_metric, context_recall_metric
	faithfulness_metric = Faithfulness(llm=llm)
	answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
	context_precision_metric = ContextPrecision(llm=llm)
	context_recall_metric = ContextRecall(llm=llm)

	# 3. 加载数据集并运行实验
	df = pd.read_csv("datasets/eval_dataset_full.csv")
	dataset = Dataset.from_pandas(df, name="eval_dataset_vector", backend="inmemory")
	semaphore = asyncio.Semaphore(3)  # 限制并发
	results = await wuxia_rag_experiment.arun(dataset, name="wuxia_eval_v1", semaphore=semaphore)

	# 4. 保存结果
	results.to_pandas().to_csv("datasets/eval_results_full.csv", index=False, encoding="utf-8-sig")
	print("ok")


asyncio.run(main())
