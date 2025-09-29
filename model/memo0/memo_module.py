

import argparse
import os

from src.langmem import LangMemManager
from src.memzero.add import MemoryADD
from src.memzero.search import MemorySearch
from src.openai.predict import OpenAIPredict
from src.rag import RAGManager
from src.utils import METHODS, TECHNIQUES
from src.zep.add import ZepAdd
from src.zep.search import ZepSearch
from openai import OpenAI
import openai

# class OpenAIClient:
#     def __init__(self, api_key, base_url):
#         self.api_key = api_key
#         self.base_url = base_url
#         openai.api_key = self.api_key
#         openai.api_base = self.base_url

#     def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):

#         response = gpt_client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content.strip()


# client = OpenAIClient(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url=os.environ.get("OPENAI_API_BASE")
# )

import os
import json

from src.memzero.add import MemoryADD
from src.memzero.search import MemorySearch

class Memo0Model:
    """
    Wrapper class for MeMo0 model.
    Provides a unified interface for pipeline evaluation.
    """
    def __init__(self, mem_dir="mem_tmp_Memo0", top_k=30, filter_memories=False, is_graph=False):
        self.mem_dir = mem_dir
        os.makedirs(self.mem_dir, exist_ok=True)
        self.top_k = top_k
        self.filter_memories = filter_memories
        self.is_graph = is_graph

    def init_memory(self, dataset_path):
        """
        Preprocess dataset and add memories (run 'add' step)
        """
        memory_manager = MemoryADD(data_path=dataset_path, is_graph=self.is_graph)
        memory_manager.process_all_conversations()
        # Store reference to memory manager for search
        self.memory_manager = memory_manager

    def generate_answer(self, idx, sample, dataset_path="dataset/locomo10.json"):
        """
        Run 'search' step for a single sample and return system answer
        """
        # Ensure memory has been initialized
        if not hasattr(self, "memory_manager"):
            self.init_memory(dataset_path)

        output_file_path = os.path.join(
            self.mem_dir,
            f"memo0_results_top_{self.top_k}_filter_{self.filter_memories}_graph_{self.is_graph}.jsonl"
        )

        memory_searcher = MemorySearch(
            output_file_path=output_file_path,
            top_k=self.top_k,
            filter_memories=self.filter_memories,
            is_graph=self.is_graph
        )

        # Process single sample
        memory_searcher.process_sample(sample)

        # Retrieve answer from searcher
        system_answer = memory_searcher.get_answer(sample)  # 需要 MemorySearch 提供 get_answer 方法
        return system_answer
