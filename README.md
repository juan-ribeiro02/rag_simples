# Estudo de RAG Simples em Python

Repositório didático para estudo prático de **RAG (Retrieval-Augmented Generation / Geração Aumentada por Recuperação)** utilizando embeddings locais, banco vetorial em memória e a API do Google Gemini.

---

## Estrutura do Projeto

O repositório está dividido em duas etapas evolutivas:

```text
rag_simples/
├── .env                                 # Chave de API do Google Gemini (GEMINI_API_KEY)
├── README.md                            # Documentação do projeto
├── v1_rag_lista/
│   └── rag_simples.py                   # Versão 1: RAG básico usando uma lista de frases fixa no código
└── v2_rag_arquivos/
    ├── rag_arquivos.py                  # Versão 2: RAG lendo arquivos reais (.txt, .md, .pdf) com chunking e metadados
    └── documentos/                      # Pasta com os documentos da base de conhecimento
        ├── edital_iniciacao_cientifica.pdf
        ├── manual_cluster_gpu.md
        └── regras_laboratorio.txt
```

---

## Tecnologias Utilizadas

- **`sentence-transformers`**: Executa o modelo de embedding `all-MiniLM-L6-v2` localmente, de graça e sem precisar de chave de API para vetorizar textos.
- **`chromadb`**: Banco de dados vetorial que roda em memória/local para indexar os vetores e buscar os trechos semanticamente mais próximos da pergunta.
- **`google-genai`**: SDK oficial do Google para enviar o prompt enriquecido com o contexto recuperado ao modelo Gemini.
- **`python-dotenv`**: Carrega a variável `GEMINI_API_KEY` a partir do arquivo `.env`.
- **`pypdf`** *(usado na v2)*: Extrai o texto de arquivos `.pdf` localmente.

---

## Como Configurar e Rodar

### 1. Instalar as dependências

```bash
pip install sentence-transformers chromadb google-genai python-dotenv pypdf
```

### 2. Configurar a chave de API

Crie um arquivo `.env` na raiz do projeto com a sua chave do Google AI Studio:

```env
GEMINI_API_KEY="sua_chave_aqui"
```

### 3. Executar a Versão 1 (Lista em Memória)

Na `v1_rag_lista`, os documentos são frases curtas declaradas diretamente numa lista Python:

```bash
python v1_rag_lista/rag_simples.py
```

### 4. Executar a Versão 2 (Leitura de Arquivos Reais)

Na `v2_rag_arquivos`, o script:
1. Lê todos os arquivos `.txt`, `.md` e `.pdf` da pasta `v2_rag_arquivos/documentos/`;
2. Divide textos longos em pedaços menores (**chunks** com sobreposição/overlap) para preservar o contexto;
3. Salva os vetores e o nome do arquivo de origem (`metadatas`) no ChromaDB;
4. Permite digitar uma pergunta no terminal (ou pressionar `Enter` para rodar a pergunta de teste) e cita a fonte na resposta final.

```bash
python v2_rag_arquivos/rag_arquivos.py
```

Você também pode adicionar novos arquivos `.txt`, `.md` ou `.pdf` dentro de `v2_rag_arquivos/documentos/` e rodar o script novamente para testá-los automaticamente.
