import os
from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

#carregar um modelo leve de embedding 
print("Carregando modelo de embedding....")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#inicializar o banco vetorial
chrome_client = chromadb.Client()
collection = chrome_client.get_or_create_collection(name="documentos_teste")

#documentos para servir de base
documentos = [
    "O Laboratório de Inovação funciona de segunda a sexta, das 08h às 18h no Bloco B.",
    "Para solicitar acesso ao cluster de GPUs, o aluno deve enviar um e-mail para o professor coordenador.",
    "O café do laboratório é comunitário, e a limpeza da cafeteira é feita às sextas-feiras.",
    "Projetos de iniciação científica recebem auxílio financeiro através do edital anual da instituição."
]

#gerando embeddings para cada documento
print("Gerando embedding e salvando no chromeDB...")
embeddings = embedding_model.encode(documentos).tolist()
ids = [f"doc_{i}" for i in range(len(documentos))]

collection.add(
    ids = ids,
    embeddings = embeddings,
    documents = documentos
)
print("Base de conhecimento indexada com sucesso.\n")

def buscar_contexto(pergunta: str, top_k: int = 2) -> list[str]:
    #faz o embedding da pergunta tbm
    pergunta_vetor = embedding_model.encode([pergunta]).tolist()

    #busca os top_k documentos mais parecidos
    resultados = collection.query(
        query_embeddings=pergunta_vetor,
        n_results=top_k
    )

    # Garante que "documents" existe e tem itens antes de pegar o índice 0
    docs = resultados.get("documents")
    if docs and len(docs) > 0:
        return docs[0]
    
    return []

#teste de busca
pergunta = "Como faço para usar as placas de vídeo do lab?"
contexto_encontrado = buscar_contexto(pergunta)

print(f"Pergunta: {pergunta}")
print("Trechos mais relevantes encontrados:")
for i, chunk in enumerate(contexto_encontrado, 1):
    print(f" [{i}] {chunk}")
print("-" * 50)

def montar_prompt(pergunta: str, contextos: list[str]) -> str:
    contexto_formatado = "\n".join([f"- {c}" for c in contextos]) 

    prompt = f"""Você é um assistente do laboratório.
    Responda à pergunta do usuário baseando-se ESTRITAMENTE no contexto fornecido abaixo.
    Se a informação não estiver no contexto, diga claramente: "Não encontrei essa informação no material do laboratório."

    CONTEXTO:
    {contexto_formatado}

    PERGUNTA:
    {pergunta}

    RESPOSTA:"""

    return prompt

prompt_final = montar_prompt(pergunta, contexto_encontrado)
print("Prompt pronto para ser enviado ao LLM:\n")
print(prompt_final)

client = genai.Client(api_key = gemini_api_key)

# Chamada ao modelo enviando o prompt com o contexto injetado
resposta = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt_final,
)

print("\n=== RESPOSTA DO LLM ===")
print(resposta.text)
