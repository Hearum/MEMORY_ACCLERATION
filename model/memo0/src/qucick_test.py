from openai import OpenAI
from mem0 import Memory
from mem0.configs.base import MemoryConfig

openai_client = OpenAI()
import os
EMBED_DIM = 384  
memory_config = MemoryConfig(
    embedder={
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",  # HF 模型名
            "embedding_dims": EMBED_DIM,
        }
    },
    vector_store={
        "provider": "faiss",
        "config": {
            "path": "./memory_index",
            "collection_name": "mem0",
            "embedding_model_dims": EMBED_DIM
        }
    },
    llm={
        "provider": "openai",          
        "config": {
        "model": "LLAMA",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "openrouter_base_url": os.environ.get("OPENAI_API_BASE")
    }
    }
)
"""
export OPENROUTER_API_KEY="nope"
export OPENROUTER_API_BASE="http://localhost:30087/v1"

"""
memory = Memory(memory_config)
# memory = Memory()


gpt_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

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

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    memory.reset()
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    import pdb
    pdb.set_trace()
    response = gpt_client.chat.completions.create(model="LLAMA", messages=messages)
    assistant_response = response.choices[0].message.content

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()