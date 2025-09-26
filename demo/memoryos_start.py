import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from memoryos import Memoryos


# --- 基本配置 ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
API_KEY = "YOUR_OPENAI_API_KEY"  # 替换为您的密钥
BASE_URL = ""  # 可选：如果使用自定义 OpenAI 端点
DATA_STORAGE_PATH = "./simple_demo_data"
LLM_MODEL = "gpt-4o-mini"

def simple_demo():
    print("MemoryOS 简单演示")

    # 1. 初始化 MemoryOS
    print("正在初始化 MemoryOS...")
    try:
        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,
            mid_term_heat_threshold=5,
            retrieval_queue_capacity=7,
            long_term_knowledge_capacity=100,
            # 支持 Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3, all-MiniLM-L6-v2
            embedding_model_name="BAAI/bge-m3"
        )
        print("MemoryOS 初始化成功！n")
    except Exception as e:
        print(f"错误: {e}")
        return

    # 2. 添加一些基本记忆
    print("正在添加一些记忆...")

    memo.add_memory(
        user_input="你好！我是汤姆，我在旧金山做数据科学家。",
        agent_response="你好汤姆！很高兴认识你。数据科学是一个非常令人兴奋的领域。你主要处理什么样的数据？"
    )

    test_query = "你记得我的工作是什么吗？"
    print(f"用户: {test_query}")

    response = memo.get_response(
        query=test_query,
    )

    print(f"助手: {response}")

if __name__ == "__main__":
    simple_demo()