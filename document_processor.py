# import os
# import time
# import logging
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.documents import Document

# logger = logging.getLogger(__name__)

# class DocumentProcessor:
#     """Handles document loading and processing with enhanced error handling."""
#     def __init__(self, config, gpu_manager):
#         self.config = config
#         self.gpu_manager = gpu_manager
#         self.error_count = 0

#     def load_documents(self):
#         if not os.path.exists(self.config.pdf_directory):
#             raise ValueError(f"Directory not found: {self.config.pdf_directory}")
#         pdf_files = []
#         for root, _, files in os.walk(self.config.pdf_directory):
#             pdf_files.extend(
#                 os.path.join(root, file)
#                 for file in files if file.lower().endswith('.pdf')
#             )
#         if not pdf_files:
#             raise ValueError(f"No PDF files found in {self.config.pdf_directory}")
#         logger.info(f"Found {len(pdf_files)} PDF files")
#         documents = []
#         failed_files = []
#         with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
#             future_to_pdf = {
#                 executor.submit(self._load_single_pdf_with_retry, pdf): pdf for pdf in pdf_files
#             }
#             for future in tqdm(future_to_pdf, desc="Loading PDFs"):
#                 try:
#                     result = future.result()
#                     if result:
#                         documents.extend(result)
#                     else:
#                         failed_files.append(future_to_pdf[future])
#                 except Exception as e:
#                     logger.error(f"Failed to process future: {str(e)}")
#                     failed_files.append(future_to_pdf[future])
#         if failed_files:
#             logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
#         return documents

#     def _load_single_pdf_with_retry(self, file_path, attempt=1):
#         try:
#             loader = PyPDFLoader(file_path)
#             documents = loader.load()
#             if not documents:
#                 raise ValueError("No content loaded from PDF")
#             return documents
#         except Exception as e:
#             if attempt < self.config.max_retries:
#                 logger.warning(f"Retry {attempt} for {file_path}: {str(e)}")
#                 time.sleep(self.config.retry_delay)
#                 return self._load_single_pdf_with_retry(file_path, attempt + 1)
#             else:
#                 logger.error(f"Failed to load PDF {file_path} after {attempt} attempts: {str(e)}")
#                 self.error_count += 1
#                 return []

#     def split_documents(self, documents):
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.config.chunk_size,
#             chunk_overlap=self.config.chunk_overlap,
#             separators=["\n\n", "\n", ". ", " ", ""],
#             length_function=len
#         )
#         splits = []
#         batch_size = self.config.batch_size
#         for i in range(0, len(documents), batch_size):
#             batch = documents[i:i + batch_size]
#             try:
#                 batch_splits = splitter.split_documents(batch)
#                 valid_splits = self._validate_splits(batch_splits)
#                 splits.extend(valid_splits)
#                 self.gpu_manager.optimize_memory()
#             except Exception as e:
#                 logger.error(f"Batch splitting failed for batch {i//batch_size}: {str(e)}")
#         logger.info(f"Created {len(splits)} quality chunks from {len(documents)} documents")
#         return splits

#     def _validate_splits(self, splits):
#         return [
#             doc for doc in splits
#             if (doc.page_content.strip() and len(doc.page_content.split()) >= 5 and 
#                 not doc.page_content.isspace() and len(doc.page_content) <= self.config.chunk_size * 1.1)
#         ]
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import psutil

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing with enhanced error handling and resource awareness."""
    
    def __init__(self, config, gpu_manager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.error_count = 0
        
        # Optionally adjust max_workers based on available CPUs.
        if not self.config.max_workers or self.config.max_workers < 1:
            self.config.max_workers = psutil.cpu_count(logical=True) or 4

    def load_documents(self):
        """Load documents using parallel processing with retry mechanism."""
        if not os.path.exists(self.config.pdf_directory):
            raise ValueError(f"Directory not found: {self.config.pdf_directory}")

        pdf_files = []
        for root, _, files in os.walk(self.config.pdf_directory):
            pdf_files.extend(
                os.path.join(root, file)
                for file in files if file.lower().endswith('.pdf')
            )
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.config.pdf_directory}")

        logger.info(f"Found {len(pdf_files)} PDF files")
        documents = []
        failed_files = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_pdf = {
                executor.submit(self._load_single_pdf_with_retry, pdf): pdf
                for pdf in pdf_files
            }
            for future in tqdm(future_to_pdf, desc="Loading PDFs"):
                try:
                    result = future.result()
                    if result:
                        documents.extend(result)
                    else:
                        failed_files.append(future_to_pdf[future])
                except Exception as e:
                    logger.error(f"Failed to process {future_to_pdf[future]}: {str(e)}")
                    failed_files.append(future_to_pdf[future])
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
        return documents

    def _load_single_pdf_with_retry(self, file_path, attempt=1):
        """Attempt to load a PDF file with a retry mechanism."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                raise ValueError("No content loaded from PDF")
            return documents
        except Exception as e:
            if attempt < self.config.max_retries:
                logger.warning(f"Retry {attempt} for {file_path}: {str(e)}")
                time.sleep(self.config.retry_delay)
                return self._load_single_pdf_with_retry(file_path, attempt + 1)
            else:
                logger.error(f"Failed to load {file_path} after {attempt} attempts: {str(e)}")
                self.error_count += 1
                return []

    def split_documents(self, documents):
        """Split documents into chunks with content validation."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        splits = []
        batch_size = self.config.batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                batch_splits = splitter.split_documents(batch)
                valid_splits = self._validate_splits(batch_splits)
                splits.extend(valid_splits)
                self.gpu_manager.optimize_memory()
            except Exception as e:
                logger.error(f"Batch splitting failed for batch {i // batch_size}: {str(e)}")
        logger.info(f"Created {len(splits)} quality chunks from {len(documents)} documents")
        return splits

    def _validate_splits(self, splits):
        """Filter out invalid or trivial document splits."""
        return [
            doc for doc in splits
            if (doc.page_content.strip() and len(doc.page_content.split()) >= 5 and 
                not doc.page_content.isspace() and len(doc.page_content) <= self.config.chunk_size * 1.1)
        ]
