"""
	运行RAG，补充（response,retrieved_contexts） 到已有的评估数据集中，
"""

import pandas as pd
from dotenv import load_dotenv

from rag import WuxiaRAGSystem

# 1.加载环境变量，初始化RAG系统
load_dotenv()
rag_system = WuxiaRAGSystem()
rag_system.init_systems_and_build_knowledge_base()

# 2. 读取你的评估数据集
df = pd.read_csv("datasets/eval_dataset.csv")

# 3. 运行 RAG 管道，补充两列
print("开始运行RAG，获取数据")
responses, retrieved_contexts = rag_system.get_rag_output_for_eval(df["user_input"].tolist())
df["response"] = responses
df["retrieved_contexts"] = retrieved_contexts

# 4. 保存补充后的数据集
df.to_csv("datasets/eval_dataset_full.csv", index=False)

print("补充数据集完成")