import logging
from dotenv import load_dotenv
import traceback
from rag import WuxiaRAGSystem, SearchType

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,          # 想看到 DEBUG 就改成 logging.DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def main():
    try:

        #test_content = "一千个头 神仙姐姐" #神仙姊姊
        #test_content = "段誉 一千个头"
        test_content = "无量山洞 一千个头"

        # 创建RAG系统
        rag_system = WuxiaRAGSystem()

        # 根据内容找到需要的块，帮助排查问题
        rag_system.test_get_chunks("解脱为乐")
        # rag_system.test_get_chunks("人生于世")

        # 测试查询
        #rag_system.test_query(test_content, SearchType.VECTOR)
        #rag_system.test_query(test_content,SearchType.BM25)
        #rag_system.test_query(test_content, SearchType.RRF)

    except Exception as e:
        logger.error(f"系统运行出错: {e}",exc_info=True)
        print(f"系统错误: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()

