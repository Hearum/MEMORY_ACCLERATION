import os
import json
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

load_dotenv()

PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGModel:
    def __init__(self, data_path="dataset/locomo10_rag.json", chunk_size=500, k=1):
        self.model = os.getenv("MODEL")
        self.client = OpenAI()
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def generate_answer(self, idx, sample, dataset_name, output_file, dataset_type="locomo"):
        """
        Args:
            idx: 样本索引
            sample: 数据样本
            dataset_name: 数据集名称
            output_file: 输出文件路径
            dataset_type: 数据集类型 (locomo/longmemeval)
        """
        question = sample["question"]
        answer = sample.get("answer", "")
        category = sample.get("category", "")

        chat_history = sample.get("conversation", [])
        chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

        if self.chunk_size == -1:
            context = chunks[0]
            search_time = 0
        else:
            context, search_time = self.search(question, chunks, embeddings, k=self.k)

        response, response_time = self.generate_response(question, context)

        result = {
            "id": idx,
            "dataset": dataset_name,
            "question": question,
            "answer": answer,
            "category": category,
            "context": context,
            "response": response,
            "search_time": search_time,
            "response_time": response_time,
        }

        # 追加写入
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return result

    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        retries = 0
        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that can answer "
                                "questions based on the provided context. "
                                "If the question involves timing, use the conversation date for reference. "
                                "Provide the shortest possible answer. "
                                "Use words directly from the conversation when possible. "
                                "Avoid using subjects in your answer."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                t2 = time.time()
                return response.choices[0].message.content.strip(), t2 - t1
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)

    def clean_chat_history(self, chat_history):
        return "\n".join(
            f"{c['timestamp']} | {c['speaker']}: {c['text']}" for c in chat_history
        )

    def calculate_embedding(self, document):
        response = self.client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"), input=document)
        return response.data[0].embedding

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        similarities = [self.calculate_similarity(query_embedding, emb) for emb in embeddings]

        if k == 1:
            top_indices = [np.argmax(similarities)]
        else:
            top_indices = np.argsort(similarities)[-k:][::-1]

        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])
        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        encoding = tiktoken.encoding_for_model(os.getenv("EMBEDDING_MODEL"))
        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents], []

        chunks, embeddings = [], []
        tokens = encoding.encode(documents)

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        for chunk in chunks:
            embeddings.append(self.calculate_embedding(chunk))

        return chunks, embeddings
