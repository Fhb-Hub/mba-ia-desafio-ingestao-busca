import os
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_SIMILAR_DOCUMENTS = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(query: str) -> str:
    try:
        if not query:
            return "Por favor, forneça uma pergunta."

        llm = _init_chat_model()
        embeddings = _initialize_embeddings_model()
        store = _initialize_vector_store(embeddings)
        similar_documents = _search_similar_documents(store, query)
        context_text = _format_context_as_string(similar_documents)
        prompt_template = _create_prompt_template()
        chain = _build_chain(prompt_template, llm)
        answer = chain.invoke({"contexto": context_text, "pergunta": query})

        return answer

    except Exception as e:
        print(f"Ocorreu um erro durante a busca: {e}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta."


def _init_chat_model() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL"), temperature=0
    )


def _initialize_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        google_api_key=GOOGLE_API_KEY,
    )


def _initialize_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def _search_similar_documents(
    store: PGVector, question: str
) -> List[Tuple[Document, float]]:
    return store.similarity_search_with_score(question, k=MAX_SIMILAR_DOCUMENTS)


def _format_context_as_string(context: List[Tuple[Document, float]]) -> str:
    if not context:
        return ""
    return "\n".join(doc.page_content.strip() for doc, _ in context)


def _create_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def _build_chain(prompt: ChatPromptTemplate, llm: BaseChatModel) -> Runnable:
    return prompt | llm | StrOutputParser()
