# Desafio MBA Engenharia de Software com IA - Full Cycle

Um sistema de **Retrieval-Augmented Generation (RAG)** construído com Python, LangChain e o poder do Google Gemini. Este projeto transforma um documento PDF em uma base de conhecimento interativa, permitindo que você "converse" com seus documentos através de um chat no terminal.

## Visão Geral

O objetivo deste projeto é demonstrar a implementação de um pipeline RAG. Ele extrai informações de um documento não estruturado (PDF), os processa em vetores de embedding e os armazena em um banco de dados vetorial para recuperação eficiente. Quando um usuário faz uma pergunta, o sistema primeiro recupera os trechos de texto mais relevantes e, em seguida, usa um Large Language Model (LLM) para gerar uma resposta coesa e contextualizada.

## Arquitetura do Pipeline RAG

O fluxo de dados segue as seguintes etapas:

1.  **Carregamento (Load):** O documento PDF é carregado a partir do caminho especificado no arquivo de ambiente.
2.  **Divisão (Split):** O texto extraído é dividido em pedaços menores (chunks) para otimizar a busca.
3.  **Embedding (Embed):** Cada chunk é transformado em um vetor numérico (embedding) usando a API do Google Gemini.
4.  **Armazenamento (Store):** Os vetores e o texto correspondente são armazenados no **PostgreSQL com a extensão pgvector**.
5.  **Recuperação (Retrieve):** Dada uma pergunta, o sistema a converte em um vetor e busca os chunks mais similares no banco de dados.
6.  **Geração (Generate):** Os chunks recuperados são enviados como contexto para o Google Gemini, que gera a resposta final.

## Tecnologias Utilizadas

-   **Linguagem:** Python 3.11+
-   **Framework de Orquestração:** [LangChain](https://www.langchain.com/)
-   **Modelo de IA (LLM):** [Google Gemini](https://ai.google.dev/)
-   **Banco de Dados Vetorial:** [PostgreSQL](https://www.postgresql.org/) com [pgvector](https://github.com/pgvector/pgvector)
-   **Containerização:** Docker e Docker Compose
-   **Principais Bibliotecas Python:** `langchain-google-genai`, `langchain-postgres`, `pypdf`, `python-dotenv`, `psycopg`.

## Começando

Siga os passos abaixo para configurar e executar o projeto localmente.

### Pré-requisitos

-   Python 3.11 ou superior
-   Docker e Docker Compose
-   Git
-   Uma chave de API do Google Gemini. Você pode criar uma no [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key).

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd mba-ia-desafio-ingestao-busca
    ```

2.  **Configure as Variáveis de Ambiente:**
    -   Renomeie o arquivo `.env.example` para `.env`.
    -   Abra o arquivo `.env` e preencha as variáveis com suas informações.
    
    ```ini
    # Chave de API para autenticação com os serviços do Google AI
    GOOGLE_API_KEY=sua_google_gemini_api_key_aqui

    # Modelo de embedding para transformar texto em vetores
    GOOGLE_EMBEDDING_MODEL='models/embedding-001'

    # Modelo de linguagem generativo para responder às perguntas
    GOOGLE_MODEL='gemini-1.5-flash'

    # String de conexão para o banco de dados PostgreSQL
    DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

    # Nome da coleção (tabela) onde os vetores serão armazenados
    COLLECTION_NAME=documents_collection

    # Caminho para o arquivo PDF que será processado
    PDF_PATH=document.pdf
    ```

    > **Nota:** O `DATABASE_URL` no `.env` está configurado para `localhost` para que a aplicação Python possa se conectar ao container Docker.

3.  **Inicie os Serviços com Docker:**
    ```bash
    docker-compose up -d
    ```

4.  **Crie o Ambiente Virtual e Instale as Dependências:**
    ```bash
    # Crie o ambiente virtual
    python -m venv venv

    # Ative o ambiente (Windows)
    venv\Scripts\activate
    # Ative o ambiente (Linux/macOS)
    source venv/bin/activate

    # Instale as bibliotecas Python
    pip install -r requirements.txt
    ```

## Uso

1.  **Adicione seu Documento:**
    -   Coloque o arquivo PDF que você deseja processar na raiz do projeto.
    -   Certifique-se de que o caminho para o arquivo PDF esteja corretamente configurado na variável `PDF_PATH` em seu arquivo `.env`.

2.  **Popule o Banco de Dados:**
    -   Execute o script `ingest.py` para ler o PDF, gerar os embeddings e salvá-los no banco de dados.
    ```bash
    python src/ingest.py
    ```

3.  **Inicie o Chat:**
    -   Execute o `chat.py` para iniciar a interface de chat no terminal.
    ```bash
    python src/chat.py
    ```
    -   Agora você pode fazer perguntas sobre o conteúdo do seu documento diretamente no terminal! Para sair, digite `sair`.

## Estrutura do Projeto

```
.
├── .env.example        # Arquivo de exemplo para variáveis de ambiente
├── .gitignore          # Arquivos ignorados pelo Git
├── docker-compose.yml  # Define os serviços Docker (PostgreSQL + pgvector)
├── document.pdf        # Documento de exemplo para ingestão
├── README.md           # Esta documentação
├── requirements.txt    # Dependências do projeto Python
└── src/
    ├── ingest.py       # Script para carregar, dividir e vetorizar o PDF, armazenando no PGVector.
    ├── search.py       # Módulo que contém a lógica RAG: busca documentos similares e gera a resposta com o LLM.
    └── chat.py         # Ponto de entrada do usuário, gerencia a interface de linha de comando.
```