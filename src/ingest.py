import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_postgres import PGVector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
PDF_PATH = os.getenv("PDF_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GOOGLE_API_KEY:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi configurada.")


def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Erro: O arquivo '{PDF_PATH}' não foi encontrado.")
        return

    documents = _load_documents(PDF_PATH)
    chunks = _split_documents_into_chunks(documents)
    enriched_chunks = _enrich_chunks(chunks)
    ids = _generate_chunks_ids(enriched_chunks)
    embeddings = _initialize_embeddings_model()
    _add_documents_to_vector_store(
        enriched_chunks,
        ids,
        embeddings,
    )


def _load_documents(pdf_path: str) -> list[Document]:
    print(f"Iniciando a ingestão do arquivo: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"O documento foi carregado. Total de {len(documents)} páginas.")
    return documents


def _split_documents_into_chunks(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"O documento foi dividido em {len(chunks)} chunks.")
    return chunks


def _enrich_document(doc: Document) -> Document:
    return Document(
        page_content=doc.page_content,
        metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)},
    )


def _enrich_chunks(chunks: list[Document]) -> list[Document]:
    return [_enrich_document(doc) for doc in chunks]


def _generate_chunks_ids(chunks: list[Document]) -> list[str]:
    return [f"doc-{i}" for i in range(len(chunks))]


def _initialize_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    print("Modelo de embeddings do Google inicializado.")
    return embeddings


def _add_documents_to_vector_store(
    chunks: list[Document],
    ids: list[str],
    embeddings,
):
    print(f"Iniciando a inserção dos vetores na coleção '{COLLECTION_NAME}'...")
    try:
        store = _initialize_vector_store(embeddings)
        store.add_documents(documents=chunks, ids=ids)
        print("Ingestão concluída com sucesso!")
        print(f"{len(chunks)} chunks foram vetorizados e armazenados no PostgreSQL.")
    except Exception as e:
        print(f"Ocorreu um erro durante a ingestão: {e}")


def _initialize_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


if __name__ == "__main__":
    ingest_pdf()
