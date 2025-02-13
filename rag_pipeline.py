# import os
# import time
# import logging
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_ollama.llms import OllamaLLM

# from gpu_manager import GPUManager
# from document_processor import DocumentProcessor
# from vector_store_manager import VectorStoreManager

# logger = logging.getLogger(__name__)

# class RAGPipeline:
#     def __init__(self, config):
#         self.config = config
#         self.gpu_manager = GPUManager(config)
#         self.doc_processor = DocumentProcessor(config, self.gpu_manager)
#         self.vector_store_manager = VectorStoreManager(config, self.gpu_manager)
#         self.start_time = None
#         self.vectorstore = None

#     def setup_chain(self):
#         retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": self.config.top_k_results}
#         )
#         llm = OllamaLLM(
#             model=self.config.llm_model,
#             temperature=self.config.temperature
#         )
#         system_prompt = """You are a knowledgeable assistant for document analysis.

# Use the following context to answer the question. When answering:
# 1. Be specific and cite relevant details from the provided context
# 2. If information is missing or unclear, explicitly state what's missing
# 3. Use direct quotes when appropriate, citing the source document
# 4. Maintain factual accuracy and avoid speculation
# 5. If multiple sources provide conflicting information, acknowledge the differences

# Context:
# {context}

# Remember to:
# - Stay within the provided context
# - Indicate if the answer is incomplete
# - Highlight any uncertainties
# """
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "{input}")
#         ])
#         chain = create_stuff_documents_chain(llm, prompt)
#         return chain, retriever

#     def process_query(self, query, chain, retriever):
#         try:
#             retrieved_docs = retriever.invoke(query)
#             if not retrieved_docs:
#                 return {"answer": "No relevant information found in the documents.", "sources": []}
#             valid_docs = [
#                 doc for doc in retrieved_docs if doc.page_content and not doc.page_content.isspace()
#             ]
#             if not valid_docs:
#                 return {"answer": "Retrieved documents were invalid or empty.", "sources": []}
#             result = chain.invoke({
#                 "context": valid_docs,
#                 "input": query
#             })
#             self.gpu_manager.optimize_memory()
#             return {
#                 "answer": result,
#                 "sources": [
#                     {
#                         "content": doc.page_content[:200],
#                         "metadata": doc.metadata,
#                         "relevance_score": getattr(doc, 'relevance_score', None)
#                     } for doc in valid_docs
#                 ]
#             }
#         except Exception as e:
#             logger.error(f"Query processing failed: {str(e)}")
#             return {"error": str(e), "answer": "An error occurred while processing your query."}

#     def run_pipeline(self):
#         try:
#             logger.info("Initializing RAG pipeline")
#             self.start_time = time.time()
#             documents = self.doc_processor.load_documents()
#             logger.info(f"Loaded {len(documents)} documents in {time.time() - self.start_time:.2f} seconds")
#             splits = self.doc_processor.split_documents(documents)
#             logger.info(f"Created {len(splits)} splits in {time.time() - self.start_time:.2f} seconds")
#             self.vectorstore = self.vector_store_manager.create_vector_store(splits)
#             logger.info(f"Created vector store in {time.time() - self.start_time:.2f} seconds")
#             chain, retriever = self.setup_chain()
#             return chain, retriever
#         except Exception as e:
#             logger.error(f"Pipeline execution failed: {str(e)}")
#             raise
#         finally:
#             end_time = time.time()
#             total_time = end_time - self.start_time
#             logger.info(f"Pipeline completed in {total_time:.2f} seconds")
#             if self.config.persist_directory and self.vectorstore:
#                 try:
#                     self.vectorstore.persist()
#                     logger.info("Vector store persisted successfully")
#                 except Exception as e:
#                     logger.error(f"Failed to persist vector store: {str(e)}")
#             self.gpu_manager.optimize_memory()
#             if hasattr(self.doc_processor, 'error_count') and self.doc_processor.error_count > 0:
#                 logger.warning(f"Total document processing errors: {self.doc_processor.error_count}")

# def main():
#     import os
#     import logging
#     from dataclasses import dataclass

#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('rag_system.log'),
#             logging.StreamHandler()
#         ]
#     )

#     @dataclass
#     class RAGConfig:
#         pdf_directory: str
#         embeddings_model: str
#         llm_model: str
#         chunk_size: int = 1000
#         chunk_overlap: int = 200
#         max_workers: int = 4
#         temperature: float = 0.7
#         top_k_results: int = 4
#         persist_directory: str = "./chroma_db"
#         batch_size: int = 32
#         use_cuda: bool = True
#         cuda_device: int = 0
#         max_retries: int = 3
#         retry_delay: float = 1.0
#         memory_buffer: float = 0.9

#     config = RAGConfig(
#         pdf_directory=os.getenv('RAG_PDF_DIR', r"D:\Books\python"),
#         embeddings_model=os.getenv('RAG_EMBEDDINGS_MODEL', "nomic-embed-text:latest"),
#         llm_model=os.getenv('RAG_LLM_MODEL', "llama3.2b:custom"),
#         chunk_size=int(os.getenv('RAG_CHUNK_SIZE', "1000")),
#         chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', "200")),
#         max_workers=int(os.getenv('RAG_MAX_WORKERS', "4")),
#         temperature=float(os.getenv('RAG_TEMPERATURE', "0.7")),
#         top_k_results=int(os.getenv('RAG_TOP_K_RESULTS', "4")),
#         persist_directory=os.getenv('RAG_PERSIST_DIR', "./chroma_db"),
#         batch_size=int(os.getenv('RAG_BATCH_SIZE', "32")),
#         use_cuda=os.getenv('RAG_USE_CUDA', "true").lower() == "true",
#         cuda_device=int(os.getenv('RAG_CUDA_DEVICE', "0")),
#         max_retries=int(os.getenv('RAG_MAX_RETRIES', "3")),
#         retry_delay=float(os.getenv('RAG_RETRY_DELAY', "1.0")),
#         memory_buffer=float(os.getenv('RAG_MEMORY_BUFFER', "0.9"))
#     )
#     pipeline = RAGPipeline(config)
#     chain, retriever = pipeline.run_pipeline()
#     # Hand off chain and retriever to the chat interface
#     from chat_interface import chat_loop
#     chat_loop(chain, retriever)

# if __name__ == "__main__":
#     main()
import os
import time
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM

from gpu_manager import GPUManager
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.gpu_manager = GPUManager(config)
        self.doc_processor = DocumentProcessor(config, self.gpu_manager)
        self.vector_store_manager = VectorStoreManager(config, self.gpu_manager)
        self.start_time = None
        self.vectorstore = None

    def setup_chain(self):
        # If vectorstore is not loaded, try to load it from disk.
        if self.vectorstore is None:
            logger.info("No vector store in memory; attempting to load from disk...")
            self.vectorstore = self.vector_store_manager.load_vector_store()
            
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k_results}
        )
        llm = OllamaLLM(
            model=self.config.llm_model,
            temperature=self.config.temperature
        )
        system_prompt = """You are a knowledgeable assistant for document analysis.

Use the following context to answer the question. When answering:
1. Be specific and cite relevant details from the provided context
2. If information is missing or unclear, explicitly state what's missing
3. Use direct quotes when appropriate, citing the source document
4. Maintain factual accuracy and avoid speculation
5. If multiple sources provide conflicting information, acknowledge the differences

Context:
{context}

Remember to:
- Stay within the provided context
- Indicate if the answer is incomplete
- Highlight any uncertainties
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        chain = create_stuff_documents_chain(llm, prompt)
        return chain, retriever

    def run_pipeline(self):
        try:
            logger.info("Initializing RAG pipeline")
            self.start_time = time.time()
            documents = self.doc_processor.load_documents()
            logger.info(f"Loaded {len(documents)} documents in {time.time() - self.start_time:.2f} seconds")
            splits = self.doc_processor.split_documents(documents)
            logger.info(f"Created {len(splits)} splits in {time.time() - self.start_time:.2f} seconds")
            self.vectorstore = self.vector_store_manager.create_vector_store(splits)
            logger.info(f"Created vector store in {time.time() - self.start_time:.2f} seconds")
            chain, retriever = self.setup_chain()
            return chain, retriever
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        finally:
            end_time = time.time()
            total_time = end_time - self.start_time
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            if self.config.persist_directory and self.vectorstore:
                try:
                    self.vectorstore.persist()
                    logger.info("Vector store persisted successfully")
                except Exception as e:
                    logger.error(f"Failed to persist vector store: {str(e)}")
            self.gpu_manager.optimize_memory()
            if hasattr(self.doc_processor, 'error_count') and self.doc_processor.error_count > 0:
                logger.warning(f"Total document processing errors: {self.doc_processor.error_count}")
