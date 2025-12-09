from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from openai import OpenAI
from openai import AzureOpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        if self.global_config.azure_embedding_endpoint is None:
            self.client = OpenAI(
                base_url=self.global_config.embedding_base_url
            )
        else:
            self.client = AzureOpenAI(api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                                      azure_endpoint=self.global_config.azure_embedding_endpoint)


    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
        results = np.array([v.embedding for v in response.data])

        return results
    import openai
    def batch_encode(self, texts: list, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        batch_size = kwargs.pop("batch_size", 16)
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 找到空文本的索引
            placeholder_indices = [j for j, t in enumerate(batch) if not t.strip()]
            valid_batch = [t for t in batch if t.strip()]

            if valid_batch:
                try:
                    embeddings = self.encode(valid_batch)
                    embedding_dim = embeddings.shape[1]  # 动态获取向量维度
                except openai.OpenAIError as e:
                    print(f"Batch {i} encoding failed: {e}")
                    import ipdb; ipdb.set_trace()
                    embedding_dim =  2560 # 默认值
                    embeddings = np.zeros((len(valid_batch), embedding_dim))
            else:
                embedding_dim = 1536  # 默认值
                embeddings = np.zeros((0, embedding_dim))

            # 插回空文本占位向量
            final_embeddings = []
            vi = 0
            for j in range(len(batch)):
                if j in placeholder_indices:
                    final_embeddings.append(np.zeros(embedding_dim))
                else:
                    final_embeddings.append(embeddings[vi])
                    vi += 1

            results.append(np.array(final_embeddings))

        return np.concatenate(results)
