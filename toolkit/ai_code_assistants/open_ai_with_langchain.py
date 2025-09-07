from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sqlalchemy import create_engine, text

OPENAI_API_KEY=""

CODE_DIR = ""
SQL_SCHEMA_FILE = "schema.sql"
VECTOR_DB_PATH = "vector_db"
DB_URL = (
)
# ------------------------------

def extract_sql_schema(db_url, output_file="schema.sql"):
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """))
        schema_text = ""
        for row in result:
            schema_text += f"{row.TABLE_NAME} | {row.COLUMN_NAME} | {row.DATA_TYPE}\n"

        with open(output_file, "w") as f:
            f.write(schema_text)
        print("✅ SQL schema written to", output_file)

def build_index():
    # Load code and schema
    code_loader = DirectoryLoader(CODE_DIR, glob="**/*.py", loader_cls=TextLoader)
    docs = code_loader.load()
    docs += TextLoader(SQL_SCHEMA_FILE).load()

    # Chunk into embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print("✅ Vector DB saved at", VECTOR_DB_PATH)


def ask_question(query):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever(search_type="similarity", k=5)
    )
    response = qa_chain.run(query)
    print("🤖:", response)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-schema", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--ask", type=str)

    args = parser.parse_args()

    if args.extract_schema:
        extract_sql_schema(DB_URL)

    if args.build:
        build_index()

    if args.ask:
        ask_question(args.ask)
