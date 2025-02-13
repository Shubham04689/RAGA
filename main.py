import os
import logging
import time
import psutil
import torch
import argparse
import subprocess
from dataclasses import dataclass
from rag_pipeline import RAGPipeline
from chat_interface import chat_loop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class RAGConfig:
    pdf_directory: str
    embeddings_model: str
    llm_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_workers: int = 0  # 0 means auto-detect
    temperature: float = 0.7
    top_k_results: int = 4
    persist_directory: str = "./chroma_db"
    batch_size: int = 32
    use_cuda: bool = True
    cuda_device: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    memory_buffer: float = 0.9

def check_system_resources(config):
    """Logs available CPU, system RAM, and GPU resources."""
    cpu_count = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    available_ram_mb = mem.available / 1024**2
    logging.info(f"System Resources: {cpu_count} CPU cores available, {available_ram_mb:.2f}MB RAM available.")

    if config.use_cuda and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(config.cuda_device)
        total_gpu_mem = torch.cuda.get_device_properties(config.cuda_device).total_memory / 1024**2
        logging.info(f"GPU Available: {device_name} with {total_gpu_mem:.2f}MB total memory.")
    else:
        logging.info("CUDA not available. Running on CPU.")

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline and Chat Interface Entry Point")
    parser.add_argument(
        "--interface",
        choices=["console", "streamlit"],
        default="console",
        help="Which chat interface to launch (default: streamlit)"
    )
    parser.add_argument(
        "--use-database",
        action="store_true",
        help="Enable vector database (ChromaDB) for context retrieval"
    )
    args = parser.parse_args()

    # Load configuration
    config = RAGConfig(
        pdf_directory=os.getenv("RAG_PDF_DIR", r"D:\Books\python"),
        embeddings_model=os.getenv("RAG_EMBEDDINGS_MODEL", "nomic-embed-text:latest"),
        llm_model=os.getenv("RAG_LLM_MODEL", "llama3.2b:custom"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        max_workers=int(os.getenv("RAG_MAX_WORKERS", "0")),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.7")),
        top_k_results=int(os.getenv("RAG_TOP_K_RESULTS", "4")),
        persist_directory=os.getenv("RAG_PERSIST_DIR", "./chroma_db"),
        batch_size=int(os.getenv("RAG_BATCH_SIZE", "32")),
        use_cuda=os.getenv("RAG_USE_CUDA", "true").lower() == "true",
        cuda_device=int(os.getenv("RAG_CUDA_DEVICE", "0")),
        max_retries=int(os.getenv("RAG_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("RAG_RETRY_DELAY", "1.0")),
        memory_buffer=float(os.getenv("RAG_MEMORY_BUFFER", "0.9"))
    )

    check_system_resources(config)

    pipeline = RAGPipeline(config)
    logging.info("Loading the existing vector store from disk...")
    chain, retriever = pipeline.setup_chain()

    if args.interface == "console":
        logging.info("Launching console chat interface...")
        chat_loop(chain, retriever)
    elif args.interface == "streamlit":
        logging.info("Launching Streamlit chat interface...")
        subprocess.run(["streamlit", "run", "streamlit_chat.py", "--", str(args.use_database)])

if __name__ == "__main__":
    main()
