from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


PRESETS = {
    "cheap_fast": {
        "llm_model": "llama-3.1-8b-instant",
        "max_tokens": 2048,
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "faiss",
        "max_papers_default": 500,
        "min_cluster_size": 5,
        "temperature": 0.3,
    },
    "balanced": {
        "llm_model": "llama-3.3-70b-versatile",
        "max_tokens": 4096,
        "embedding_model": "FremyCompany/BioLORD-2023",
        "vector_db": "faiss",
        "max_papers_default": 2000,
        "min_cluster_size": 10,
        "temperature": 0.4,
    },
    "max_quality": {
        "llm_model": "llama-3.3-70b-versatile",
        "max_tokens": 8192,
        "embedding_model": "FremyCompany/BioLORD-2023",
        "vector_db": "faiss",
        "max_papers_default": 5000,
        "min_cluster_size": 15,
        "temperature": 0.5,
    },
}


def get_preset(preset_name: str) -> dict:
    """Return preset configuration by name, defaulting to 'balanced'."""
    if preset_name not in PRESETS:
        preset_name = "balanced"
    return PRESETS[preset_name]


class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    PINECONE_API_KEY: str = Field(default="", env="PINECONE_API_KEY")
    PUBMED_EMAIL: str = Field(default="researcher@example.com", env="PUBMED_EMAIL")
    LLM_PRESET: str = Field(default="balanced", env="LLM_PRESET")
    VECTOR_DB: str = Field(default="faiss", env="VECTOR_DB")
    CACHE_DIR: str = Field(default=".cache", env="CACHE_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_preset(self) -> dict:
        return get_preset(self.LLM_PRESET)


settings = Settings()
