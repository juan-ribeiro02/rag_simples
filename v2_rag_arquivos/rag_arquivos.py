import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Carrega as variáveis de ambiente (.env na raiz do projeto)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Diretório onde ficam os arquivos reais (.txt, .md, .pdf)
PASTA_DOCUMENTOS = Path(__file__).parent / "documentos"


def extrair_texto_arquivo(caminho: Path) -> str:
    """Lê o conteúdo de um arquivo .txt, .md ou .pdf e retorna como string."""
    extensao = caminho.suffix.lower()

    if extensao in [".txt", ".md"]:
        return caminho.read_text(encoding="utf-8")

    elif extensao == ".pdf":
        leitor = PdfReader(str(caminho))
        paginas = [pagina.extract_text() or "" for pagina in leitor.pages]
        return "\n".join(paginas)

    return ""


def dividir_em_chunks(texto: str, tamanho_chunk: int = 450, sobreposicao: int = 90) -> list[str]:
    """
    Divide um texto longo em pedaços menores (chunks) com sobreposição (overlap),
    respeitando os espaços entre palavras para não cortar palavras ao meio.
    """
    texto_limpo = " ".join(texto.split())
    if not texto_limpo:
        return []

    chunks = []
    inicio = 0
    tamanho_total = len(texto_limpo)

    while inicio < tamanho_total:
        fim = min(inicio + tamanho_chunk, tamanho_total)

        # Se não chegou ao final do texto, corta no último espaço em branco
        if fim < tamanho_total:
            ultimo_espaco = texto_limpo.rfind(" ", inicio, fim)
            if ultimo_espaco > inicio:
                fim = ultimo_espaco

        trecho = texto_limpo[inicio:fim].strip()
        if trecho:
            chunks.append(trecho)

        if fim >= tamanho_total:
            break

        # Avança voltando 'sobreposicao' caracteres e alinha no próximo espaço
        # para não começar o próximo chunk no meio de uma palavra
        proximo_inicio = max(inicio + 1, fim - sobreposicao)
        espaco_inicio = texto_limpo.find(" ", proximo_inicio)
        if espaco_inicio != -1 and espaco_inicio < fim:
            inicio = espaco_inicio + 1
        else:
            inicio = proximo_inicio

    return chunks


def carregar_documentos_da_pasta(pasta: Path) -> tuple[list[str], list[dict], list[str]]:
    """
    Percorre a pasta de documentos, lê os arquivos suportados (.txt, .md, .pdf),
    divide em chunks e prepara as listas para o ChromaDB.
    """
    todos_chunks = []
    todos_metadados = []
    todos_ids = []

    extensoes_suportadas = {".txt", ".md", ".pdf"}
    arquivos = sorted([f for f in pasta.iterdir() if f.is_file() and f.suffix.lower() in extensoes_suportadas])

    for arquivo in arquivos:
        print(f" -> Lendo arquivo: {arquivo.name}")
        texto = extrair_texto_arquivo(arquivo)
        chunks = dividir_em_chunks(texto)

        for i, chunk in enumerate(chunks):
            todos_chunks.append(chunk)
            todos_metadados.append({"fonte": arquivo.name, "chunk_id": i})
            todos_ids.append(f"{arquivo.stem}_chunk_{i}")

    return todos_chunks, todos_metadados, todos_ids


# 1. Carregar modelo leve de embedding local
print("Carregando modelo de embedding....")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Inicializar o banco vetorial em memória
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="documentos_arquivos")

# 3. Ler os arquivos reais da pasta e dividir em chunks
print(f"\nCarregando arquivos da pasta '{PASTA_DOCUMENTOS.name}/'...")
documentos_chunks, metadados, ids = carregar_documentos_da_pasta(PASTA_DOCUMENTOS)

# 4. Gerar embeddings e salvar no ChromaDB junto com os metadados (nome do arquivo)
print(f"\nGerando embeddings para {len(documentos_chunks)} trechos (chunks) e salvando no ChromaDB...")
embeddings = embedding_model.encode(documentos_chunks).tolist()

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documentos_chunks,
    metadatas=metadados
)
print("Base de conhecimento de arquivos indexada com sucesso!\n")


def buscar_contexto(pergunta: str, top_k: int = 3) -> list[dict]:
    """Busca os top_k trechos mais relevantes junto com o nome do arquivo de origem."""
    pergunta_vetor = embedding_model.encode([pergunta]).tolist()

    resultados = collection.query(
        query_embeddings=pergunta_vetor,
        n_results=top_k
    )

    docs = resultados.get("documents", [[]])[0]
    metas = resultados.get("metadatas", [[]])[0]

    contextos = []
    for trecho, meta in zip(docs, metas):
        contextos.append({
            "fonte": meta.get("fonte", "desconhecido"),
            "texto": trecho
        })

    return contextos


def montar_prompt(pergunta: str, contextos: list[dict]) -> str:
    contexto_formatado = "\n".join(
        [f"- [Fonte: {c['fonte']}] {c['texto']}" for c in contextos]
    )

    prompt = f"""Você é um assistente do laboratório.
Responda à pergunta do usuário baseando-se ESTRITAMENTE no contexto fornecido abaixo.
Cite o nome do arquivo de origem (Fonte) na sua resposta quando utilizar a informação.
Se a informação não estiver no contexto, diga claramente: "Não encontrei essa informação no material do laboratório."

CONTEXTO:
{contexto_formatado}

PERGUNTA:
{pergunta}

RESPOSTA:"""

    return prompt


# 5. Teste de busca (permite digitar uma pergunta ou apertar Enter para usar a padrão)
pergunta_padrao = "Como faço para usar as placas de vídeo do lab e qual o limite de tempo por execução?"
pergunta_usuario = input(f"Digite sua pergunta (ou pressione Enter para usar: '{pergunta_padrao}'): ").strip()
pergunta = pergunta_usuario if pergunta_usuario else pergunta_padrao

contexto_encontrado = buscar_contexto(pergunta, top_k=3)

print(f"\nPergunta: {pergunta}")
print("Trechos mais relevantes encontrados nos arquivos:")
for i, item in enumerate(contexto_encontrado, 1):
    print(f" [{i}] ({item['fonte']}) {item['texto']}")
print("-" * 60)

prompt_final = montar_prompt(pergunta, contexto_encontrado)
print("Prompt pronto para ser enviado ao LLM:\n")
print(prompt_final)

# 6. Chamada ao LLM (Google Gemini)
client = genai.Client(api_key=gemini_api_key)

resposta = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt_final,
)

print("\n=== RESPOSTA DO LLM ===")
print(resposta.text)
