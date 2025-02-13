import logging
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations with enhanced error handling."""
    def __init__(self, config, gpu_manager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.embeddings = OllamaEmbeddings(model=config.embeddings_model)

    def create_vector_store(self, splits):
        logger.info("Initializing vector store")
        valid_splits = []
        failed_chunks = []
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(splits), batch_size), desc="Processing document batches"):
            batch = splits[i:i + batch_size]
            try:
                embeddings = self._process_batch_with_validation(batch)
                if embeddings:
                    valid_splits.extend(batch)
                else:
                    failed_chunks.extend(batch)
                self.gpu_manager.optimize_memory()
            except Exception as e:
                logger.error(f"Batch {i//batch_size} processing failed: {str(e)}")
                failed_chunks.extend(batch)
        if not valid_splits:
            raise ValueError("No valid documents after embedding validation")
        if failed_chunks:
            logger.warning(f"Failed to process {len(failed_chunks)} chunks")
        vectorstore = Chroma.from_documents(
            documents=valid_splits,
            embedding=self.embeddings,
            persist_directory=self.config.persist_directory
        )
        return vectorstore

    def _process_batch_with_validation(self, batch):
        try:
            embeddings = self.embeddings.embed_documents([doc.page_content for doc in batch])
            return bool(embeddings and all(isinstance(emb, list) for emb in embeddings))
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            return False

    def load_vector_store(self):
        """
        Loads an existing vector store from disk.
        Assumes that the vector store was previously persisted in the given persist_directory.
        """
        try:
            vectorstore = Chroma(
                persist_directory=self.config.persist_directory,
                embedding_function=self.embeddings  # Pass the function directly.
            )
            logger.info("Vector store loaded from disk successfully.")
            return vectorstore
        except Exception as e:
            logger.error("Failed to load vector store from disk: " + str(e))
            raise

