import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import requests
from InstructorEmbedding import INSTRUCTOR
from sqlalchemy import text

from data_engineering.database import db_functions as database

# === Configuration ===
PROJECT_CODE_DIR = (
    "C:\\Users\\menon\\OneDrive\\Documents\\SourceCode\\InvestmentManagement"
)
VECTOR_DB_PATH = "vector_db/faiss.index"
DOC_META_PATH = "vector_db/documents.json"
CACHE_PATH = "vector_db/cache.json"
CONVERSATION_HISTORY_PATH = "vector_db/conversation_history.json"

# === Ollama Configuration ===
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:14b"

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".github",
    "vector_db",
    ".git",
    "node_modules",
    "build",
    "dist",
}
EXCLUDE_FILES = {"*.pyc", "*.pyo", "*.pyd", ".DS_Store", "Thumbs.db"}

# === Optimized Configuration ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
EMBEDDING_BATCH_SIZE = 32
MAX_CONTEXT_LENGTH = 8000  # Tokens for context window
MAX_WORKERS = 4  # For parallel processing
CONVERSATION_MEMORY_SIZE = 10  # Number of previous exchanges to remember

# Set up logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rag_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Track query performance metrics"""

    query_time: float
    embedding_time: float
    search_time: float
    generation_time: float
    num_chunks_retrieved: int
    context_length: int


class ConversationMemory:
    """Manage conversation history for context-aware responses"""

    def __init__(self, max_size: int = CONVERSATION_MEMORY_SIZE):
        self.max_size = max_size
        self.history = []
        self.load_history()

    def add_exchange(self, query: str, response: str):
        """Add a query-response pair to history"""
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
            }
        )

        # Keep only recent exchanges
        if len(self.history) > self.max_size:
            self.history = self.history[-self.max_size :]

        self.save_history()

    def get_context_string(self) -> str:
        """Get conversation history as context string"""
        if not self.history:
            return ""

        context_parts = []
        for exchange in self.history[-3:]:  # Last 3 exchanges
            context_parts.append(f"Previous Q: {exchange['query']}")
            context_parts.append(f"Previous A: {exchange['response'][:200]}...")

        return "\n".join(context_parts)

    def load_history(self):
        """Load conversation history from file"""
        if os.path.exists(CONVERSATION_HISTORY_PATH):
            try:
                with open(CONVERSATION_HISTORY_PATH, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
                self.history = []

    def save_history(self):
        """Save conversation history to file"""
        try:
            os.makedirs(os.path.dirname(CONVERSATION_HISTORY_PATH), exist_ok=True)
            with open(CONVERSATION_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save conversation history: {e}")


class EmbeddingManager:
    """Singleton pattern for embedding model management"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._embedder = None
        return cls._instance

    def get_embedder(self):
        if self._embedder is None:
            logger.info("Loading embedding model...")
            self._embedder = INSTRUCTOR("hkunlp/instructor-base")
            logger.info("Embedding model loaded successfully!")
        return self._embedder


# Global embedding manager
embedding_manager = EmbeddingManager()


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        # Configure session for better performance
        self.session.headers.update(
            {"Connection": "keep-alive", "Content-Type": "application/json"}
        )

        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def warm_up_model(self) -> bool:
        """Warm up the model with a simple query to reduce first-query latency"""
        try:
            logger.info(f"Warming up model {self.model}...")
            simple_prompt = "Hello, ready to help with code!"
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": simple_prompt,
                    "stream": False,
                    "options": {"num_predict": 10},
                },
                timeout=60,
            )
            if response.status_code == 200:
                logger.info("Model warmed up successfully!")
                return True
            return False
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
            return False

    def check_model_availability(self) -> bool:
        """Check if the model is available in Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return any(
                    self.model.split(":")[0] in model for model in available_models
                )
            return False
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    def pull_model(self) -> bool:
        """Pull the model if it's not available"""
        try:
            logger.info(f"Pulling model {self.model}...")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=600,  # 10 minutes for model download
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                print(f"\r{data['status']}", end="", flush=True)
                            if data.get("status") == "success":
                                print("\n✅ Model pulled successfully!")
                                return True
                        except json.JSONDecodeError:
                            continue
            return False
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def generate_with_retry(self, payload: dict, max_retries: int = 3) -> Optional[str]:
        """Generate response with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=300,  # 5 minutes
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {response.status_code}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Enhanced chat with better error handling and optimization"""
        try:
            # Try chat endpoint first
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                },
            }

            response = self.session.post(
                f"{self.base_url}/api/chat", json=payload, timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                # Fallback to generate endpoint
                prompt = self._messages_to_prompt(messages)
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                    },
                }

                result = self.generate_with_retry(payload)
                return (
                    result
                    if result
                    else "Sorry, I'm having trouble generating a response right now."
                )

        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return f"Error: Unable to generate response - {str(e)}"

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt with better formatting"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")

        return "".join(prompt_parts) + "<|assistant|>\n"


def extract_functions_and_classes(
    code: str, file_path: str = ""
) -> List[Tuple[str, str, Dict]]:
    """Enhanced extraction with better metadata"""
    chunks = []

    # Extract imports for context
    import_lines = []
    for line in code.split("\n")[:20]:  # Check first 20 lines
        if line.strip().startswith(("import ", "from ")):
            import_lines.append(line.strip())

    imports_context = "\n".join(import_lines) if import_lines else ""

    # Extract functions with better regex
    func_pattern = r"((?:@\w+(?:\([^)]*\))?\s*\n)*)\s*def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?:[^{]*?(?=\n(?:def|class|\Z|@\w+\s*\ndef|@\w+\s*\nclass))"
    for match in re.finditer(func_pattern, code, re.DOTALL | re.MULTILINE):
        decorators = match.group(1).strip() if match.group(1) else ""
        func_name = match.group(2)
        func_code = match.group(0).strip()

        if len(func_code) > MIN_CHUNK_SIZE:
            # Add imports context for standalone understanding
            full_code = (
                f"{imports_context}\n\n{func_code}" if imports_context else func_code
            )

            chunks.append(
                (
                    f"Function: {func_name}",
                    full_code,
                    {
                        "entity_type": "function",
                        "entity_name": func_name,
                        "has_decorators": bool(decorators),
                        "file_path": file_path,
                        "line_count": len(func_code.split("\n")),
                    },
                )
            )

    # Extract classes with methods
    class_pattern = r"((?:@\w+(?:\([^)]*\))?\s*\n)*)\s*class\s+(\w+)(?:\([^)]*\))?:[^{]*?(?=\n(?:def|class|\Z|@\w+\s*\ndef|@\w+\s*\nclass))"
    for match in re.finditer(class_pattern, code, re.DOTALL | re.MULTILINE):
        decorators = match.group(1).strip() if match.group(1) else ""
        class_name = match.group(2)
        class_code = match.group(0).strip()

        if len(class_code) > MIN_CHUNK_SIZE:
            # Count methods in the class
            method_count = len(re.findall(r"\n\s+def\s+\w+", class_code))

            full_code = (
                f"{imports_context}\n\n{class_code}" if imports_context else class_code
            )

            chunks.append(
                (
                    f"Class: {class_name}",
                    full_code,
                    {
                        "entity_type": "class",
                        "entity_name": class_name,
                        "has_decorators": bool(decorators),
                        "method_count": method_count,
                        "file_path": file_path,
                        "line_count": len(class_code.split("\n")),
                    },
                )
            )

    return chunks


def smart_chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Enhanced smart chunking with better boundary detection"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_size = 0

    # Identify logical boundaries (function/class definitions, comments, etc.)
    logical_boundaries = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith(("def ", "class ", "# ===", '"""', "'''"))
            or stripped.endswith(('"""', "'''"))
            or (stripped.startswith("#") and len(stripped) > 10)
        ):
            logical_boundaries.add(i)

    for i, line in enumerate(lines):
        line_size = len(line) + 1

        if current_size + line_size > chunk_size and current_chunk:
            # Try to break at a logical boundary if we're close
            if i in logical_boundaries or any(
                abs(i - b) <= 3 for b in logical_boundaries
            ):
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)

                # Smart overlap: include relevant context
                overlap_lines = []
                overlap_size = 0
                for prev_line in reversed(current_chunk):
                    if overlap_size + len(prev_line) + 1 <= overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_size += len(prev_line) + 1
                    else:
                        break

                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return [chunk for chunk in chunks if len(chunk) > MIN_CHUNK_SIZE]


def load_code_chunks_parallel(path: str, exclude_dirs: set) -> List[Dict]:
    """Parallel code loading for better performance"""
    chunks = []
    file_extensions = {
        ".py",
        ".js",
        ".ts",
        ".sql",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".html",
        ".css",
    }

    # Collect all files first
    files_to_process = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for fname in files:
            if any(fname.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, fname)
                relative_path = os.path.relpath(file_path, path)
                files_to_process.append((file_path, relative_path, fname))

    def process_file(file_info):
        file_path, relative_path, fname = file_info
        file_chunks = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read().strip()

            if len(code) < MIN_CHUNK_SIZE:
                return file_chunks

            # File-level metadata
            file_stats = os.stat(file_path)
            base_metadata = {
                "file_type": fname.split(".")[-1],
                "file_size": len(code),
                "file_path": relative_path,
                "last_modified": datetime.fromtimestamp(
                    file_stats.st_mtime
                ).isoformat(),
                "line_count": len(code.split("\n")),
            }

            # Semantic extraction for Python files
            if fname.endswith(".py"):
                semantic_chunks = extract_functions_and_classes(code, relative_path)
                for chunk_type, chunk_code, metadata in semantic_chunks:
                    combined_metadata = {**base_metadata, **metadata}
                    combined_metadata["chunk_type"] = "semantic"

                    file_chunks.append(
                        {
                            "source": f"{relative_path}:{metadata['entity_name']}",
                            "type": "code_semantic",
                            "content": chunk_code,
                            "metadata": combined_metadata,
                        }
                    )

            # Text chunking for all files
            text_chunks = smart_chunk_text(code)
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "chunk_type": "text",
                    }
                )

                file_chunks.append(
                    {
                        "source": f"{relative_path}:chunk_{i}",
                        "type": "code_text",
                        "content": chunk,
                        "metadata": chunk_metadata,
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")

        return file_chunks

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_file, file_info): file_info
            for file_info in files_to_process
        }

        for future in as_completed(future_to_file):
            file_chunks = future.result()
            chunks.extend(file_chunks)

    logger.info(f"Loaded {len(chunks)} code chunks from {len(files_to_process)} files")
    return chunks


def introspect_sql_schema_enhanced() -> List[Dict]:
    """Enhanced SQL schema introspection with relationships and constraints"""
    engine, connection, session = database.get_db_connection()
    chunks = []

    try:
        with connection:
            # Get table schemas with enhanced information
            schema_query = text(
                """
                SELECT      c.TABLE_SCHEMA,
                           c.TABLE_NAME,
                           c.COLUMN_NAME,
                           c.ORDINAL_POSITION,
                           c.COLUMN_DEFAULT,
                           c.IS_NULLABLE,
                           c.DATA_TYPE,
                           c.CHARACTER_MAXIMUM_LENGTH,
                           c.NUMERIC_PRECISION,
                           c.NUMERIC_SCALE,
                           kcu.CONSTRAINT_NAME,
                           tc.CONSTRAINT_TYPE
                FROM        INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN   INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                           ON c.TABLE_SCHEMA = kcu.TABLE_SCHEMA 
                           AND c.TABLE_NAME = kcu.TABLE_NAME 
                           AND c.COLUMN_NAME = kcu.COLUMN_NAME
                LEFT JOIN   INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc 
                           ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                WHERE       c.TABLE_NAME NOT IN ('database_firewall_rules', 'sysdiagrams')
                ORDER BY    c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
            """
            )

            result = connection.execute(schema_query)
            table_docs = {}

            for row in result:
                schema_name = row.TABLE_SCHEMA
                table_name = row.TABLE_NAME
                full_table_name = f"{schema_name}.{table_name}"

                if full_table_name not in table_docs:
                    table_docs[full_table_name] = {
                        "columns": [],
                        "constraints": set(),
                        "primary_keys": [],
                        "foreign_keys": [],
                    }

                # Build column info
                type_info = row.DATA_TYPE
                if row.CHARACTER_MAXIMUM_LENGTH:
                    type_info += f"({row.CHARACTER_MAXIMUM_LENGTH})"
                elif row.NUMERIC_PRECISION and row.NUMERIC_SCALE is not None:
                    type_info += f"({row.NUMERIC_PRECISION},{row.NUMERIC_SCALE})"
                elif row.NUMERIC_PRECISION:
                    type_info += f"({row.NUMERIC_PRECISION})"

                nullable_str = "NULL" if row.IS_NULLABLE == "YES" else "NOT NULL"
                default_str = (
                    f" DEFAULT {row.COLUMN_DEFAULT}" if row.COLUMN_DEFAULT else ""
                )

                constraint_info = ""
                if row.CONSTRAINT_TYPE == "PRIMARY KEY":
                    constraint_info = " [PK]"
                    table_docs[full_table_name]["primary_keys"].append(row.COLUMN_NAME)
                elif row.CONSTRAINT_TYPE == "FOREIGN KEY":
                    constraint_info = " [FK]"
                    table_docs[full_table_name]["foreign_keys"].append(row.COLUMN_NAME)

                col_info = f"{row.COLUMN_NAME} {type_info} {nullable_str}{default_str}{constraint_info}"
                table_docs[full_table_name]["columns"].append(col_info)

            # Create enhanced schema chunks
            for full_table_name, table_info in table_docs.items():
                schema_parts = [f"Table: {full_table_name}"]

                if table_info["primary_keys"]:
                    schema_parts.append(
                        f"Primary Keys: {', '.join(table_info['primary_keys'])}"
                    )

                if table_info["foreign_keys"]:
                    schema_parts.append(
                        f"Foreign Keys: {', '.join(table_info['foreign_keys'])}"
                    )

                schema_parts.append("Columns:")
                schema_parts.extend(f"  {col}" for col in table_info["columns"])

                schema_text = "\n".join(schema_parts)

                chunks.append(
                    {
                        "source": f"schema_{full_table_name.replace('.', '_')}",
                        "type": "sql_schema",
                        "content": schema_text,
                        "metadata": {
                            "table_name": full_table_name,
                            "column_count": len(table_info["columns"]),
                            "has_primary_key": bool(table_info["primary_keys"]),
                            "has_foreign_keys": bool(table_info["foreign_keys"]),
                            "schema_type": "table_definition",
                        },
                    }
                )

            # Get sample data with better error handling
            for full_table_name in table_docs:
                try:
                    schema_name, table_name = full_table_name.split(".", 1)
                    sample_result = connection.execute(
                        text(f"SELECT TOP 5 * FROM [{schema_name}].[{table_name}]")
                    )
                    rows = sample_result.fetchall()

                    if rows:
                        col_names = list(sample_result.keys())
                        sample_parts = [
                            f"Sample data from {full_table_name}:",
                            f"Columns: {', '.join(col_names)}",
                            "",
                        ]

                        for i, row in enumerate(rows, 1):
                            row_dict = dict(zip(col_names, row))
                            # Truncate long values for readability
                            truncated_row = {
                                k: (str(v)[:50] + "..." if len(str(v)) > 50 else v)
                                for k, v in row_dict.items()
                            }
                            sample_parts.append(f"Row {i}: {truncated_row}")

                        sample_text = "\n".join(sample_parts)

                        chunks.append(
                            {
                                "source": f"sample_{full_table_name.replace('.', '_')}",
                                "type": "sql_sample",
                                "content": sample_text,
                                "metadata": {
                                    "table_name": full_table_name,
                                    "sample_rows": len(rows),
                                    "schema_type": "sample_data",
                                },
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to get sample data for {full_table_name}: {e}"
                    )
                    continue

    except Exception as e:
        logger.error(f"Database introspection failed: {e}")

    logger.info(f"Loaded {len(chunks)} database chunks")
    return chunks


def compute_content_hash(chunks: List[Dict]) -> str:
    """Enhanced content hashing with metadata"""
    content_data = {
        "chunks": [
            {"content": chunk["content"], "type": chunk["type"]} for chunk in chunks
        ],
        "timestamp": datetime.now().isoformat(),
        "chunk_count": len(chunks),
    }
    content_str = json.dumps(content_data, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def build_faiss_index_optimized(chunks: List[Dict], output_path: str, meta_path: str):
    """Optimized index building with progress tracking and better error handling"""
    content_hash = compute_content_hash(chunks)
    cache = load_cache()

    if cache.get("content_hash") == content_hash and os.path.exists(output_path):
        logger.info("Using cached index - no changes detected")
        return

    logger.info(f"Building optimized index for {len(chunks)} chunks...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    embedder = embedding_manager.get_embedder()
    instruction = "Represent the code/database content for semantic search: "

    all_embeddings = []
    failed_chunks = []

    # Process in batches with progress tracking
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        batch_texts = []
        batch_indices = []

        for j, chunk in enumerate(batch):
            try:
                # Truncate very long content to prevent embedding issues
                content = chunk["content"]
                if len(content) > 8000:  # Rough token limit
                    content = content[:8000] + "\n... [truncated]"

                batch_texts.append(content)
                batch_indices.append(i + j)
            except Exception as e:
                logger.warning(f"Failed to prepare chunk {i + j}: {e}")
                failed_chunks.append(i + j)
                continue

        if batch_texts:
            try:
                batch_embeddings = embedder.encode(
                    [[instruction, text] for text in batch_texts]
                )
                all_embeddings.extend(batch_embeddings)

                progress = min(i + EMBEDDING_BATCH_SIZE, len(chunks))
                logger.info(
                    f"Embedded {progress}/{len(chunks)} chunks ({progress/len(chunks)*100:.1f}%)"
                )

            except Exception as e:
                logger.error(f"Failed to embed batch starting at {i}: {e}")
                failed_chunks.extend(batch_indices)
                continue

    if failed_chunks:
        logger.warning(f"Failed to process {len(failed_chunks)} chunks")

    if not all_embeddings:
        raise ValueError("No embeddings were generated successfully")

    embeddings = np.array(all_embeddings).astype("float32")
    logger.info(f"Created embeddings matrix: {embeddings.shape}")

    # Choose index type based on data size and dimensionality
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    if num_vectors > 10000:
        # Use IVF for large datasets
        nlist = min(int(np.sqrt(num_vectors)), 1000)
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        logger.info(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(nlist // 4, 50)  # Search 25% of clusters
        logger.info(f"IVF index built with nprobe={index.nprobe}")
    elif num_vectors > 1000:
        # Use HNSW for medium datasets (better recall than IVF)
        index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections per node
        index.hnsw.efConstruction = 200  # Higher = better quality, slower build
        index.add(embeddings)
        index.hnsw.efSearch = 64  # Higher = better recall, slower search
        logger.info("HNSW index built")
    else:
        # Use flat index for small datasets
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info("Flat L2 index built")

    # Save index
    faiss.write_index(index, output_path)
    logger.info(f"Index saved to {output_path}")

    # Save metadata with enhanced information
    metadata = {
        "chunks": chunks,
        "build_info": {
            "timestamp": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "failed_chunks": len(failed_chunks),
            "embedding_dimension": dimension,
            "index_type": type(index).__name__,
            "content_hash": content_hash,
        },
        "failed_indices": failed_chunks,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Update cache
    cache.update(
        {
            "content_hash": content_hash,
            "build_timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks),
        }
    )
    save_cache(cache)

    logger.info(f"Enhanced index built successfully!")


def load_cache() -> Dict:
    """Load cache with better error handling"""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
                # Validate cache structure
                if isinstance(cache, dict):
                    return cache
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return {}


def save_cache(cache_data: Dict):
    """Save cache with atomic write"""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        # Atomic write using temporary file
        temp_path = CACHE_PATH + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, CACHE_PATH)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def load_faiss_index(
    index_path: str, meta_path: str
) -> Tuple[faiss.Index, List[Dict], Dict]:
    """Load FAISS index and metadata with validation"""
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Loaded index: {type(index).__name__} with {index.ntotal} vectors")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Handle both old and new metadata formats
        if isinstance(metadata, list):
            # Old format - just chunks
            chunks = metadata
            build_info = {}
        else:
            # New format - structured metadata
            chunks = metadata.get("chunks", [])
            build_info = metadata.get("build_info", {})

        logger.info(f"Loaded {len(chunks)} chunk metadata")
        return index, chunks, build_info

    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        raise


def advanced_rerank_results(
    query: str, results: List[Tuple[float, Dict]], k: int = 5
) -> List[Dict]:
    """Advanced reranking with multiple signals"""
    if not results:
        return []

    query_lower = query.lower()
    reranked = []

    # Define scoring weights
    weights = {
        "base_similarity": 1.0,
        "query_type_match": 0.15,
        "recency_bonus": 0.05,
        "semantic_bonus": 0.1,
        "length_penalty": 0.05,
    }

    # Detect query intent
    is_function_query = any(
        kw in query_lower for kw in ["function", "def", "method", "call"]
    )
    is_class_query = any(kw in query_lower for kw in ["class", "object", "inherit"])
    is_db_query = any(
        kw in query_lower for kw in ["table", "column", "database", "sql", "select"]
    )
    is_how_query = query_lower.startswith(("how", "what", "why", "when", "where"))

    for distance, chunk in results:
        # Base similarity score (inverse of distance)
        base_score = 1.0 / (1.0 + distance)

        # Query type matching bonus
        type_bonus = 0.0
        chunk_type = chunk.get("type", "")
        metadata = chunk.get("metadata", {})

        if is_function_query and chunk_type == "code_semantic":
            if metadata.get("entity_type") == "function":
                type_bonus += 0.2
        elif is_class_query and chunk_type == "code_semantic":
            if metadata.get("entity_type") == "class":
                type_bonus += 0.2
        elif is_db_query and chunk_type.startswith("sql"):
            type_bonus += 0.15

        # Recency bonus for recently modified files
        recency_bonus = 0.0
        if "last_modified" in metadata:
            try:
                mod_time = datetime.fromisoformat(metadata["last_modified"])
                days_old = (datetime.now() - mod_time).days
                if days_old < 7:
                    recency_bonus = 0.1
                elif days_old < 30:
                    recency_bonus = 0.05
            except:
                pass

        # Semantic chunk bonus for code-related queries
        semantic_bonus = 0.0
        if chunk_type == "code_semantic" and not is_db_query:
            semantic_bonus = 0.1

        # Length penalty for very short or very long chunks
        length_penalty = 0.0
        content_length = len(chunk.get("content", ""))
        if content_length < 200:
            length_penalty = -0.05
        elif content_length > 5000:
            length_penalty = -0.03

        # Calculate final score
        final_score = (
            base_score * weights["base_similarity"]
            + type_bonus * weights["query_type_match"]
            + recency_bonus * weights["recency_bonus"]
            + semantic_bonus * weights["semantic_bonus"]
            + length_penalty * weights["length_penalty"]
        )

        reranked.append((final_score, chunk, distance))

    # Sort by final score and return top k
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk, _ in reranked[:k]]


def format_context_with_metadata(
    chunks: List[Dict], max_length: int = MAX_CONTEXT_LENGTH
) -> str:
    """Format context with better organization and metadata"""
    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        content = chunk["content"]
        chunk_type = chunk["type"]
        metadata = chunk.get("metadata", {})

        # Create header with metadata
        header_parts = [f"Source {i}: {source} ({chunk_type})"]

        if metadata.get("entity_type"):
            header_parts.append(f"Entity: {metadata['entity_type']}")
        if metadata.get("file_type"):
            header_parts.append(f"Type: {metadata['file_type']}")
        if metadata.get("line_count"):
            header_parts.append(f"Lines: {metadata['line_count']}")

        header = f"[{' | '.join(header_parts)}]"

        # Truncate content if needed
        content_to_add = content
        estimated_length = (
            len(header) + len(content_to_add) + 50
        )  # Buffer for formatting

        if current_length + estimated_length > max_length:
            remaining_space = max_length - current_length - len(header) - 100
            if remaining_space > 100:
                content_to_add = content[:remaining_space] + "\n... [truncated]"
            else:
                break  # No more space

        context_part = f"{header}\n{content_to_add}\n"
        context_parts.append(context_part)
        current_length += len(context_part)

    return "\n" + "=" * 60 + "\n" + "\n".join(context_parts)


def enhanced_query_engine(
    user_query: str,
    index: faiss.Index,
    chunks: List[Dict],
    ollama_client: OllamaClient,
    conversation_memory: ConversationMemory,
    k: int = 15,
) -> Tuple[str, QueryMetrics]:
    """Enhanced query engine with performance tracking and conversation memory"""

    start_time = time.time()

    # Embedding phase
    embed_start = time.time()
    embedder = embedding_manager.get_embedder()
    instruction = (
        "Represent the query for retrieving relevant code and database information: "
    )

    # Enhance query with conversation context
    enhanced_query = user_query
    conversation_context = conversation_memory.get_context_string()
    if conversation_context:
        enhanced_query = f"Previous context:\n{conversation_context}\n\nCurrent question: {user_query}"

    query_vec = embedder.encode([[instruction, enhanced_query]])[0].astype("float32")
    embed_time = time.time() - embed_start

    # Search phase
    search_start = time.time()
    D, I = index.search(np.array([query_vec]), k)
    search_time = time.time() - search_start

    # Get results with distances
    results = [
        (D[0][i], chunks[I[0][i]]) for i in range(len(I[0])) if I[0][i] < len(chunks)
    ]

    # Advanced reranking
    top_chunks = advanced_rerank_results(user_query, results, k=min(8, len(results)))

    # Format context
    context = format_context_with_metadata(top_chunks)
    context_length = len(context)

    # Generation phase
    gen_start = time.time()

    # Create enhanced system prompt
    system_prompt = """You are an expert coding assistant and database specialist with deep knowledge of software architecture, design patterns, and data modeling. 

Your capabilities include:
- Analyzing code structure, functions, classes, and design patterns
- Explaining database schemas, relationships, and query optimization  
- Providing implementation guidance and best practices
- Debugging and troubleshooting assistance
- Code review and improvement suggestions

Guidelines:
- Provide accurate, specific answers based on the provided context
- Include relevant code examples from the context when helpful
- Mention source files when referencing specific implementations
- If the context lacks sufficient information, clearly state this
- For database questions, explain relationships and data flow
- For code questions, explain logic and suggest improvements when appropriate
- Be concise but thorough in your explanations"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Context Information:
{context}

User Question: {user_query}

Please provide a comprehensive answer based on the context above. If you reference specific code or database elements, mention their source files.""",
        },
    ]

    # Generate response
    response = ollama_client.chat(messages, max_tokens=1500, temperature=0.7)
    gen_time = time.time() - gen_start

    total_time = time.time() - start_time

    # Create metrics
    metrics = QueryMetrics(
        query_time=total_time,
        embedding_time=embed_time,
        search_time=search_time,
        generation_time=gen_time,
        num_chunks_retrieved=len(top_chunks),
        context_length=context_length,
    )

    return response, metrics


def print_performance_stats(metrics: QueryMetrics):
    """Print detailed performance statistics"""
    print(f"\n📊 Performance Metrics:")
    print(f"   Total time: {metrics.query_time:.2f}s")
    print(f"   ├─ Embedding: {metrics.embedding_time:.3f}s")
    print(f"   ├─ Search: {metrics.search_time:.3f}s")
    print(f"   └─ Generation: {metrics.generation_time:.2f}s")
    print(f"   Retrieved chunks: {metrics.num_chunks_retrieved}")
    print(f"   Context length: {metrics.context_length:,} chars")


def interactive_mode(
    index: faiss.Index,
    chunks: List[Dict],
    build_info: Dict,
    ollama_client: OllamaClient,
):
    """Enhanced interactive mode with better UX"""
    conversation_memory = ConversationMemory()

    print("✅ Enhanced RAG system ready!")
    print(
        f"📚 Loaded {len(chunks)} chunks from {build_info.get('timestamp', 'unknown time')}"
    )
    print(f"🧠 Using {build_info.get('index_type', 'unknown')} index")
    print("\n💡 Tips:")
    print("   - Ask about specific functions, classes, or database tables")
    print("   - Use 'stats' to see system information")
    print("   - Use 'clear' to reset conversation memory")
    print("   - Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("🤔 You: ").strip()

            if user_input.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation_memory = ConversationMemory()
                print("🧹 Conversation memory cleared!")
                continue

            if user_input.lower() == "stats":
                print(f"\n📈 System Statistics:")
                print(f"   Index type: {build_info.get('index_type', 'Unknown')}")
                print(f"   Total chunks: {len(chunks):,}")
                print(f"   Built: {build_info.get('timestamp', 'Unknown')}")
                print(f"   Failed chunks: {build_info.get('failed_chunks', 0)}")
                print(
                    f"   Conversation history: {len(conversation_memory.history)} exchanges"
                )
                continue

            if not user_input:
                continue

            print("🔍 Searching and generating response...")

            answer, metrics = enhanced_query_engine(
                user_input, index, chunks, ollama_client, conversation_memory
            )

            print(f"\n🤖 Assistant: {answer}")
            print_performance_stats(metrics)
            print()

            # Add to conversation memory
            conversation_memory.add_exchange(user_input, answer)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except requests.exceptions.ReadTimeout:
            print("⏰ Request timed out. Try a simpler or shorter query.\n")
        except Exception as e:
            logger.error(f"Query error: {e}")
            print(f"❌ Error: {e}\n")


def main():
    """Enhanced main function with better error handling and progress tracking"""
    print("🚀 Starting Enhanced RAG System with Ollama...")

    try:
        # Initialize Ollama client
        ollama_client = OllamaClient()

        # Check model availability
        print(f"🔍 Checking if {OLLAMA_MODEL} is available...")
        if not ollama_client.check_model_availability():
            print(f"📥 Model {OLLAMA_MODEL} not found. Pulling model...")
            if not ollama_client.pull_model():
                print("❌ Failed to pull model. Please check your Ollama installation.")
                return
        else:
            print(f"✅ Model {OLLAMA_MODEL} is available!")

        # Warm up the model
        print("🔥 Warming up model...")
        ollama_client.warm_up_model()

        # Load or build index
        if os.path.exists(VECTOR_DB_PATH) and os.path.exists(DOC_META_PATH):
            print("📖 Loading existing index...")
            index, chunks, build_info = load_faiss_index(VECTOR_DB_PATH, DOC_META_PATH)
        else:
            print("🔨 Building new index...")

            # Load data with progress tracking
            print("📊 Loading SQL schema...")
            sql_chunks = introspect_sql_schema_enhanced()

            print("📁 Loading code files...")
            code_chunks = load_code_chunks_parallel(PROJECT_CODE_DIR, EXCLUDE_DIRS)

            all_chunks = code_chunks + sql_chunks

            print(f"\n📈 Data Summary:")
            print(f"   Code chunks: {len(code_chunks):,}")
            print(f"   SQL chunks: {len(sql_chunks):,}")
            print(f"   Total chunks: {len(all_chunks):,}")

            # Build index
            build_faiss_index_optimized(all_chunks, VECTOR_DB_PATH, DOC_META_PATH)
            index, chunks, build_info = load_faiss_index(VECTOR_DB_PATH, DOC_META_PATH)

        # Start interactive mode
        interactive_mode(index, chunks, build_info, ollama_client)

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        raise


if __name__ == "__main__":
    main()
