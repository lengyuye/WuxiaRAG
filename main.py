import logging
from dotenv import load_dotenv
import traceback
from rag import WuxiaRAGSystem

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,          # 想看到 DEBUG 就改成 logging.DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def main():
    try:
        # 创建RAG系统
        rag_system = WuxiaRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}",exc_info=True)
        print(f"系统错误: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()

