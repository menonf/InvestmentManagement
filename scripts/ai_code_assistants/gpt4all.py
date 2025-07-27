import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Tuple

import faiss
import numpy as np
from gpt4all import GPT4All
from InstructorEmbedding import INSTRUCTOR
from sqlalchemy import text

from data_engineering.database import db_functions as database

# === Configuration ===
PROJECT_CODE_DIR = "C:\\Users\\menon\\OneDrive\\Documents\\SourceCode\\InvestmentManagement"
VECTOR_DB_PATH = "vector_db/faiss.index"
DOC_META_PATH = "vector_db/documents.json"  # Changed to JSON for better metadata
CACHE_PATH = "vector_db/cache.json"
#MODEL_PATH = "C:\\Users\\menon\\AppData\\Local\\nomic.ai\\GPT4All\\Meta-Llama-3-8B-Instruct.Q4_0.gguf"

EXCLUDE_DIRS = {'.venv', '__pycache__', '.github', 'vector_db', '.git', 'node_modules'}

# === Optimized Configuration ===
CHUNK_SIZE = 1000  # Increased from 800 for better context
CHUNK_OVERLAP = 200  # 20% overlap for context continuity
MIN_CHUNK_SIZE = 100  # Skip very small chunks
EMBEDDING_BATCH_SIZE = 32  # Process embeddings in batches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Global embedder instance (avoid reloading) ===
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = INSTRUCTOR('hkunlp/instructor-base')
    return embedder

# === Enhanced Functions ===

def extract_functions_and_classes(code: str) -> List[Tuple[str, str]]:
    """Extract functions and classes as separate semantic chunks"""
    chunks = []
    
    # Extract functions
    func_pattern = r'def\s+(\w+)\s*\([^)]*\):[^{]*?(?=\n(?:def|class|\Z))'
    for match in re.finditer(func_pattern, code, re.DOTALL | re.MULTILINE):
        func_name = match.group(1)
        func_code = match.group(0).strip()
        if len(func_code) > MIN_CHUNK_SIZE:
            chunks.append((f"Function: {func_name}", func_code))
    
    # Extract classes
    class_pattern = r'class\s+(\w+)[^:]*:[^{]*?(?=\n(?:def|class|\Z))'
    for match in re.finditer(class_pattern, code, re.DOTALL | re.MULTILINE):
        class_name = match.group(1)
        class_code = match.group(0).strip()
        if len(class_code) > MIN_CHUNK_SIZE:
            chunks.append((f"Class: {class_name}", class_code))
    
    return chunks

def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Smart chunking that respects code structure"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        # If adding this line would exceed chunk size and we have content
        if current_size + line_size > chunk_size and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
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
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return [chunk for chunk in chunks if len(chunk) > MIN_CHUNK_SIZE]

def load_code_chunks(path: str, exclude_dirs: set) -> List[Dict]:
    """Enhanced code loading with better chunking and metadata"""
    chunks = []
    file_extensions = {'.py', '.js', '.ts', '.sql', '.md', '.txt', '.json', '.yaml', '.yml'}
    
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for fname in files:
            if any(fname.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, fname)
                relative_path = os.path.relpath(file_path, path)
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read().strip()
                        
                    if len(code) < MIN_CHUNK_SIZE:
                        continue
                    
                    # Try semantic chunking for Python files first
                    if fname.endswith('.py'):
                        semantic_chunks = extract_functions_and_classes(code)
                        for chunk_type, chunk_code in semantic_chunks:
                            chunks.append({
                                'source': relative_path,
                                'type': 'code_semantic',
                                'content': chunk_code,
                                'metadata': {
                                    'file_type': 'python',
                                    'semantic_type': chunk_type,
                                    'file_size': len(code)
                                }
                            })
                    
                    # Always add smart-chunked version for full context
                    text_chunks = smart_chunk_text(code)
                    for i, chunk in enumerate(text_chunks):
                        chunks.append({
                            'source': f"{relative_path}:chunk_{i}",
                            'type': 'code_text',
                            'content': chunk,
                            'metadata': {
                                'file_type': fname.split('.')[-1],
                                'chunk_index': i,
                                'total_chunks': len(text_chunks),
                                'file_size': len(code)
                            }
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue
    
    logger.info(f"Loaded {len(chunks)} code chunks from {path}")
    return chunks

def introspect_sql_schema() -> List[Dict]:
    """Enhanced SQL schema introspection with better organization"""
    engine, connection, session = database.get_db_connection()
    chunks = []

    try:
        with connection:
            # Get table schemas with schema name
            result = connection.execute(text("""
                SELECT      TABLE_SCHEMA
                           ,TABLE_NAME
                           ,COLUMN_NAME
                           ,ORDINAL_POSITION
                           ,COLUMN_DEFAULT
                           ,IS_NULLABLE
                           ,DATA_TYPE
                           ,CHARACTER_MAXIMUM_LENGTH
                           ,CHARACTER_OCTET_LENGTH
                           ,NUMERIC_PRECISION
                           ,NUMERIC_SCALE
                           ,DATETIME_PRECISION
                FROM        INFORMATION_SCHEMA.COLUMNS
                where		TABLE_NAME NOT in('database_firewall_rules','sysdiagrams')
                ORDER BY    TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """))
            
            table_docs = {}
            table_schemas = {}  # Track schema for each table
            
            for row in result:
                schema_name = row.TABLE_SCHEMA
                table_name = row.TABLE_NAME
                column_name = row.COLUMN_NAME
                data_type = row.DATA_TYPE
                is_nullable = row.IS_NULLABLE
                column_default = row.COLUMN_DEFAULT
                
                # Create full table identifier
                full_table_name = f"{schema_name}.{table_name}"
                table_schemas[table_name] = schema_name
                
                # Build column info
                nullable_str = "NULL" if is_nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {column_default}" if column_default else ""
                
                # Add precision/length info for relevant types
                type_info = data_type
                if row.CHARACTER_MAXIMUM_LENGTH:
                    type_info += f"({row.CHARACTER_MAXIMUM_LENGTH})"
                elif row.NUMERIC_PRECISION and row.NUMERIC_SCALE is not None:
                    type_info += f"({row.NUMERIC_PRECISION},{row.NUMERIC_SCALE})"
                elif row.NUMERIC_PRECISION:
                    type_info += f"({row.NUMERIC_PRECISION})"
                
                col_info = f"{column_name} {type_info} {nullable_str}{default_str}"
                table_docs.setdefault(full_table_name, []).append(col_info)

            # Create schema chunks
            for full_table_name, columns in table_docs.items():
                schema_text = f"Table: {full_table_name}\nColumns:\n" + "\n".join(f"  {col}" for col in columns)
                chunks.append({
                    'source': f"schema_{full_table_name.replace('.', '_')}",
                    'type': 'sql_schema', 
                    'content': schema_text,
                    'metadata': {
                        'table_name': full_table_name,
                        'column_count': len(columns)
                    }
                })

            # Get sample data using proper schema.table format
            for full_table_name in table_docs:
                try:
                    # Split schema and table, then bracket each separately
                    schema_name, table_name = full_table_name.split('.', 1)
                    sample_result = connection.execute(text(f"SELECT TOP 100 * FROM [{schema_name}].[{table_name}]"))
                    rows = sample_result.fetchall()
                    if rows:
                        col_names = list(sample_result.keys())
                        sample_text = f"Sample data from {full_table_name}:\n"
                        sample_text += f"Columns: {', '.join(col_names)}\n"
                        for row in rows:
                            sample_text += f"Row: {dict(zip(col_names, row))}\n"
                        
                        chunks.append({
                            'source': f"sample_{full_table_name.replace('.', '_')}",
                            'type': 'sql_sample',
                            'content': sample_text,
                            'metadata': {
                                'table_name': full_table_name,
                                'sample_rows': len(rows)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Failed to get sample data for {full_table_name}: {e}")
                    continue
                
    except Exception as e:
        logger.error(f"Database introspection failed: {e}")
    
    logger.info(f"Loaded {len(chunks)} database chunks")
    return chunks

def compute_content_hash(chunks: List[Dict]) -> str:
    """Compute hash of all content for cache invalidation"""
    content_str = json.dumps([chunk['content'] for chunk in chunks], sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()

def load_cache() -> Dict:
    """Load cache if it exists"""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache_data: Dict):
    """Save cache data"""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f)

def build_faiss_index(chunks: List[Dict], output_path: str, meta_path: str):
    """Enhanced index building with caching and batch processing"""
    content_hash = compute_content_hash(chunks)
    cache = load_cache()
    
    # Check if we can use cached embeddings
    if cache.get('content_hash') == content_hash and os.path.exists(output_path):
        logger.info("Using cached index")
        return
    
    logger.info(f"Building index for {len(chunks)} chunks...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    embedder = get_embedder()
    instruction = "Represent the code/data for retrieval: "
    
    # Process embeddings in batches for memory efficiency
    all_embeddings = []
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        batch_texts = [chunk['content'] for chunk in batch]
        batch_embeddings = embedder.encode([[instruction, text] for text in batch_texts])
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Processed {min(i + EMBEDDING_BATCH_SIZE, len(chunks))}/{len(chunks)} embeddings")
    
    embeddings = np.array(all_embeddings).astype("float32")
    
    # Build FAISS index with IVF for better performance on large datasets
    if len(chunks) > 1000:
        nlist = min(100, len(chunks) // 10)  # Number of clusters
        quantizer = faiss.IndexFlatL2(embeddings.shape[1])
        index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 10  # Search 10 clusters
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    
    faiss.write_index(index, output_path)

    # Save metadata as JSON for better structure
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Update cache
    cache['content_hash'] = content_hash
    save_cache(cache)
    
    logger.info(f"Index built and saved to {output_path}")

def load_faiss_index(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load FAISS index and metadata"""
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def rerank_results(query: str, results: List[Tuple[float, Dict]], k: int = 5) -> List[Dict]:
    """Simple reranking based on content type and relevance"""
    # Prefer semantic chunks for code queries
    if any(keyword in query.lower() for keyword in ['function', 'class', 'method', 'def']):
        semantic_bonus = 0.1
    else:
        semantic_bonus = 0.0
    
    # Rerank based on distance and type
    reranked = []
    for distance, chunk in results:
        score = 1.0 / (1.0 + distance)  # Convert distance to similarity
        if chunk['type'] == 'code_semantic':
            score += semantic_bonus
        reranked.append((score, chunk))
    
    # Sort by score and return top k
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in reranked[:k]]
def query_engine(user_query: str, index: faiss.Index, chunks: List[Dict], k: int = 10) -> str:
    """Enhanced query engine with reranking and better context formatting"""
    embedder = get_embedder()
    instruction = "Represent the query for retrieval: "
    
    # Embed query
    query_vec = embedder.encode([[instruction, user_query]])[0].astype("float32")
    
    # Search
    D, I = index.search(np.array([query_vec]), k)
    
    # Get results with distances
    results = [(D[0][i], chunks[I[0][i]]) for i in range(len(I[0])) if I[0][i] < len(chunks)]
    
    # Rerank results
    top_chunks = rerank_results(user_query, results, k=5)
    
    # Format context with better structure
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        source = chunk['source']
        content = chunk['content']
        chunk_type = chunk['type']
        
        context_part = f"[Source {i}: {source} ({chunk_type})]\n{content}\n"
        context_parts.append(context_part)
    
    context = "\n" + "="*50 + "\n".join(context_parts)
    
    # Enhanced prompt
    full_prompt = f"""You are a helpful coding assistant. Answer the question based on the provided code and database context.

Context:
{context}

Question: {user_query}

Instructions:
- Provide specific, accurate answers based on the context
- Include relevant code examples when applicable
- Mention the source files when referencing specific code
- If the context doesn't contain enough information, say so

Answer:"""

    # Generate response
    model = GPT4All(MODEL_PATH, device='NVIDIA GeForce RTX 4060 Laptop GPU', allow_download=False)
    try:
        with model.chat_session():
            response = model.generate(full_prompt, max_tokens=1024, temp=0.7)
            return response.strip()
    finally:
        model.close()

# === Main Execution ===

def main():
    print("🔍 Building enhanced vector index...")
    
    # Load data
    sql_chunks = introspect_sql_schema()
    code_chunks = load_code_chunks(PROJECT_CODE_DIR, EXCLUDE_DIRS)
    all_chunks = code_chunks + sql_chunks
    
    print(f"📊 Total chunks: {len(all_chunks)}")
    print(f"   - Code chunks: {len(code_chunks)}")
    print(f"   - SQL chunks: {len(sql_chunks)}")
    
    # Build index
    build_faiss_index(all_chunks, VECTOR_DB_PATH, DOC_META_PATH)
    index, chunks = load_faiss_index(VECTOR_DB_PATH, DOC_META_PATH)
    
    print("✅ Enhanced RAG system ready!")
    print("💡 Tips:")
    print("   - Ask about specific functions, classes, or database tables")
    print("   - Use detailed queries for better results")
    print("   - Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        
        if not user_input:
            continue
            
        print("🤔 Thinking...")
        try:
            answer = query_engine(user_input, index, chunks)
            print(f"\n🤖 Assistant: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()