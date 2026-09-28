import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Configurações gerais: modelos disponíveis, cores por categoria e exemplos de busca
ARQUIVO_BASE = Path(__file__).parent / "manifestacoes.json"
MODELOS = {
    "Multilíngue MiniLM (rápido)": "paraphrase-multilingual-MiniLM-L12-v2",
    "Multilíngue mpnet (mais preciso)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "all-MiniLM-L6-v2 (inglês, para comparação)": "sentence-transformers/all-MiniLM-L6-v2",
}
CORES_CATEGORIA = {
    "infraestrutura": "#e76f51",
    "saúde": "#2a9d8f",
    "segurança": "#264653",
    "educação": "#e9c46a",
    "meio ambiente": "#6a994e",
}
EXEMPLOS_BUSCA = [
    "rua cheia de buracos perto do mercado",
    "não tem médico no postinho",
    "som alto do bar não deixa dormir",
    "meu filho está sem merenda",
    "assalto na parada de ônibus",
]


# Carregamento com cache: modelo com cache_resource e dados/embeddings com cache_data
@st.cache_data
def carregar_base():
    with open(ARQUIVO_BASE, encoding="utf-8") as arquivo:
        base = pd.DataFrame(json.load(arquivo))
    base["tamanho"] = base["texto"].str.len()
    return base


@st.cache_resource(show_spinner="Carregando o modelo de embeddings...")
def carregar_modelo(nome_modelo):
    return SentenceTransformer(nome_modelo)


@st.cache_data(show_spinner="Gerando embeddings...")
def gerar_embeddings(nome_modelo, textos):
    return carregar_modelo(nome_modelo).encode(list(textos), normalize_embeddings=True)


@st.cache_data(show_spinner="Projetando em 2D...")
def projetar_2d(nome_modelo, textos, metodo):
    vetores = gerar_embeddings(nome_modelo, textos)
    if metodo == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(vetores)
    perplexidade = min(10, len(textos) - 1)
    return TSNE(n_components=2, perplexity=perplexidade, random_state=42).fit_transform(vetores)


# Funções auxiliares de exibição
def faixa_de_score(score):
    if score > 0.7:
        return "🟢", "alta"
    if score > 0.5:
        return "🟡", "média"
    return "🔴", "baixa"


def sugerir_categoria(resultados):
    pesos = (resultados.assign(peso=resultados["score"].clip(lower=0))
             .groupby("categoria_oficial")["peso"].sum().sort_values(ascending=False))
    total = pesos.sum()
    return pesos.index[0], (pesos.iloc[0] / total if total > 0 else 0.0)


def heatmap(matriz, rotulos, titulo, cmap="YlOrRd", anotar=False):
    tamanho = max(5, len(rotulos) * 0.3)
    fig, ax = plt.subplots(figsize=(tamanho + 1.5, tamanho))
    imagem = ax.imshow(matriz, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(rotulos)))
    ax.set_yticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=90, fontsize=7)
    ax.set_yticklabels(rotulos, fontsize=7)
    if anotar:
        for i in range(len(rotulos)):
            for j in range(len(rotulos)):
                ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if matriz[i, j] > 0.6 else "black")
    fig.colorbar(imagem, ax=ax, fraction=0.046, label="Similaridade de cosseno")
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def trecho_repetido(a, b):
    for tamanho in range(min(len(a), len(b)), 4, -1):
        if a.endswith(b[:tamanho]):
            return b[:tamanho]
    return ""


def usar_exemplo(texto):
    st.session_state["consulta"] = texto


# Aba 1: busca semântica com top-k, cores por faixa de score e sugestão de categoria
def aba_busca(base, nome_modelo, emb_base, top_k, limiar_dup):
    st.subheader("Descreva o problema com suas palavras")
    st.caption("O sistema procura manifestações com o mesmo **significado**, mesmo que usem palavras diferentes.")

    colunas = st.columns(len(EXEMPLOS_BUSCA))
    for coluna, exemplo in zip(colunas, EXEMPLOS_BUSCA):
        coluna.button(exemplo, on_click=usar_exemplo, args=(exemplo,), use_container_width=True)
    consulta = st.text_area("Manifestação", key="consulta", height=90,
                            placeholder="ex.: a rua da minha casa está toda esburacada")

    if not consulta.strip():
        st.info("Digite uma descrição ou clique em um dos exemplos acima.")
        return

    vetor = gerar_embeddings(nome_modelo, (consulta,))[0]
    resultados = base.assign(score=emb_base @ vetor).sort_values("score", ascending=False).head(top_k)

    melhor = resultados.iloc[0]
    categoria, confianca = sugerir_categoria(resultados)
    c1, c2, c3 = st.columns(3)
    c1.metric("Melhor score", f"{melhor['score']:.3f}")
    c2.metric("Categoria sugerida", categoria, f"{confianca:.0%} dos votos", delta_color="off")
    c3.metric("Resultados exibidos", len(resultados))
    if melhor["score"] >= limiar_dup:
        st.warning(f"⚠️ Possível **duplicata** de {melhor['id']} (score ≥ {limiar_dup:.2f}). "
                   "Verifique antes de abrir uma nova manifestação.")

    st.markdown(f"#### 🏆 As {len(resultados)} manifestações mais similares")
    for posicao, (_, linha) in enumerate(resultados.iterrows(), start=1):
        emoji, nivel = faixa_de_score(linha["score"])
        with st.container(border=True):
            st.markdown(f"{emoji} **#{posicao} · {linha['id']}** · {linha['categoria_oficial']} · "
                        f"{linha['data']} · score **{linha['score']:.3f}** (similaridade {nivel})")
            st.progress(float(max(0.0, min(1.0, linha["score"]))))
            st.write(linha["texto"])
    st.caption("🟢 score > 0,7 · 🟡 score > 0,5 · 🔴 demais")


# Aba 2: base completa com filtros e matriz de similaridade sob demanda
def aba_base(base, emb_base, limiar_dup):
    st.subheader("Base completa de manifestações")
    c1, c2 = st.columns([2, 3])
    categorias = c1.multiselect("Categorias", list(CORES_CATEGORIA), default=list(CORES_CATEGORIA))
    termo = c2.text_input("Filtrar por palavra no texto", placeholder="ex.: escola")

    filtrada = base[base["categoria_oficial"].isin(categorias)]
    if termo.strip():
        filtrada = filtrada[filtrada["texto"].str.contains(termo.strip(), case=False)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Manifestações exibidas", len(filtrada))
    m2.metric("Textos longos (> 500 caracteres)", int((filtrada["tamanho"] > 500).sum()))
    m3.metric("Tamanho médio", f"{filtrada['tamanho'].mean():.0f} caracteres" if len(filtrada) else "—")
    st.dataframe(filtrada[["id", "data", "categoria_oficial", "tamanho", "texto"]],
                 hide_index=True, use_container_width=True)

    if st.button("🔢 Gerar matriz de similaridade"):
        sim = cosine_similarity(emb_base)
        st.pyplot(heatmap(sim, base["id"].tolist(), "Similaridade de cosseno entre as manifestações"))

        pares = [(base["id"][i], base["id"][j], sim[i, j])
                 for i in range(len(base)) for j in range(i + 1, len(base)) if sim[i, j] >= limiar_dup]
        st.markdown(f"#### Pares acima do limiar de duplicata ({limiar_dup:.2f})")
        if pares:
            tabela = pd.DataFrame(pares, columns=["Manifestação A", "Manifestação B", "Similaridade"])
            st.dataframe(tabela.sort_values("Similaridade", ascending=False), hide_index=True)
        else:
            st.info("Nenhum par acima do limiar. Reduza o limiar na barra lateral para ver mais pares.")


# Aba 3: espaço vetorial em 2D, colorido pela categoria oficial
def aba_espaco(base, nome_modelo, emb_base, metodo):
    st.subheader(f"Espaço semântico das manifestações — {metodo}")
    coords = projetar_2d(nome_modelo, tuple(base["texto"]), metodo)
    pontos = base.assign(x=coords[:, 0], y=coords[:, 1])

    escala = alt.Scale(domain=list(CORES_CATEGORIA), range=list(CORES_CATEGORIA.values()))
    circulos = alt.Chart(pontos).mark_circle(size=160, stroke="black", strokeWidth=0.6, opacity=0.9).encode(
        x=alt.X("x", title="Dimensão 1"),
        y=alt.Y("y", title="Dimensão 2"),
        color=alt.Color("categoria_oficial", title="Categoria oficial", scale=escala),
        tooltip=["id", "categoria_oficial", "texto"],
    )
    rotulos = alt.Chart(pontos).mark_text(dx=12, dy=-8, fontSize=10).encode(x="x", y="y", text="id")
    st.altair_chart((circulos + rotulos).properties(height=520).interactive(), use_container_width=True)
    st.caption("Passe o mouse sobre um ponto para ler a manifestação. Use a roda do mouse para dar zoom.")

    # Medidas no espaço original (a projeção 2D distorce distâncias)
    categorias = base["categoria_oficial"].to_numpy()
    sim = cosine_similarity(emb_base)
    np.fill_diagonal(sim, -1)
    vizinho_mesma = categorias[sim.argmax(axis=1)] == categorias
    silhueta = silhouette_score(emb_base, categorias, metric="cosine")

    c1, c2 = st.columns(2)
    c1.metric("Vizinho mais próximo é da mesma categoria", f"{vizinho_mesma.mean():.0%}")
    c2.metric("Silhueta por categoria (−1 a 1)", f"{silhueta:.2f}")
    por_categoria = (pd.DataFrame({"Categoria": categorias, "Acerto": vizinho_mesma})
                     .groupby("Categoria")["Acerto"].mean().sort_values(ascending=False))
    st.bar_chart(por_categoria, horizontal=True, y_label="", x_label="Vizinho mais próximo na mesma categoria")

    st.markdown("#### 🔎 Os clusters semânticos coincidem com as categorias oficiais?")
    st.markdown(f"""
- **Em boa parte, sim.** Em {vizinho_mesma.mean():.0%} das manifestações, o texto mais parecido pertence à mesma
  categoria oficial. Com o MiniLM e o PCA, **saúde** e **educação** formam grupos bem separados, cada um num canto do
  gráfico: são temas com vocabulário próprio (posto, médico, consulta; escola, merenda, professor).
- **Infraestrutura, segurança e meio ambiente se misturam** no mesmo lado do gráfico, e a silhueta de {silhueta:.2f}
  (perto de 0) confirma que esses grupos se tocam. O modelo agrupa pelo **assunto concreto** do texto, enquanto as
  categorias oficiais são **administrativas**. O buraco da Av. Brasil (M003, infraestrutura) fica ao lado das
  denúncias de lixo em terrenos (M014, M023 e M035, meio ambiente); as câmeras desligadas nos postes (M021, segurança)
  ficam junto do alagamento da Rua Projetada (M020, infraestrutura). Todos são problemas "da rua", descritos com
  palavras parecidas.
- **Conclusão:** os embeddings são um bom ponto de partida para a triagem automática, principalmente em temas com
  vocabulário próprio. Nas fronteiras entre infraestrutura, segurança e meio ambiente, a categoria deve ser confirmada
  por um atendente.
""")


# Aba 4: chunking de manifestações longas, com embeddings e busca entre os chunks
def aba_chunking(base, nome_modelo):
    st.subheader("Divida uma manifestação longa em chunks")
    longas = base[base["tamanho"] > 500].sort_values("tamanho", ascending=False)
    opcoes = [f"{l['id']} · {l['categoria_oficial']} ({l['tamanho']} caracteres)" for _, l in longas.iterrows()]
    escolha = st.selectbox("Texto de exemplo", opcoes + ["✍️ Colar meu próprio texto"])
    padrao = "" if escolha.startswith("✍️") else longas.iloc[opcoes.index(escolha)]["texto"]
    texto = st.text_area("Manifestação longa", value=padrao, height=170, key=f"texto_{escolha}")

    c1, c2, c3 = st.columns(3)
    estrategia = c1.selectbox("Estratégia", ["RecursiveCharacter", "Fixed-Size (Character)"])
    chunk_size = c2.slider("chunk_size (caracteres)", 100, 800, 350, 25)
    chunk_overlap = c3.slider("chunk_overlap (caracteres)", 0, 300, 180, 10)
    st.caption("No LangChain, o overlap repete **pedaços inteiros** (frases, no caso do RecursiveCharacter). "
               "Se o overlap for menor que as frases do texto, nenhum trecho é repetido.")

    if chunk_overlap >= chunk_size:
        st.error("O overlap precisa ser menor que o chunk_size.")
        return
    if not texto.strip():
        st.info("Escolha um exemplo ou cole um texto para gerar os chunks.")
        return

    if estrategia == "RecursiveCharacter":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                  separators=["\n\n", "\n", ". ", ", ", " ", ""],
                                                  keep_separator="end")
    else:
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ")
    chunks = [c.strip() for c in splitter.split_text(texto) if c.strip()]

    vetores = gerar_embeddings(nome_modelo, tuple(chunks))
    vetor_texto = gerar_embeddings(nome_modelo, (texto,))[0]
    consecutivas = [float(vetores[i] @ vetores[i + 1]) for i in range(len(chunks) - 1)]
    cortadas = sum(not c.endswith((".", "!", "?")) for c in chunks)
    com_overlap = sum(bool(trecho_repetido(a, b)) for a, b in zip(chunks, chunks[1:]))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Chunks gerados", len(chunks))
    m2.metric("Dimensão dos embeddings", vetores.shape[1])
    m3.metric("Coesão consecutiva média", f"{np.mean(consecutivas):.3f}" if consecutivas else "—")
    m4.metric("Pares com trecho repetido", f"{com_overlap} de {max(len(chunks) - 1, 0)}")
    m5.metric("Chunks com frase cortada", f"{cortadas} de {len(chunks)}")

    st.markdown("#### 📚 Chunks gerados")
    for n, (chunk, vetor) in enumerate(zip(chunks, vetores), start=1):
        with st.expander(f"Chunk {n} · {len(chunk)} caracteres · similaridade com o texto inteiro: "
                         f"{float(vetor @ vetor_texto):.3f}"):
            st.write(chunk)
            st.caption(f"Primeiros valores do embedding: {np.round(vetor[:6], 3).tolist()} …")

    if len(chunks) < 2:
        st.info("Com um único chunk não há matriz de similaridade nem projeção para mostrar. Reduza o chunk_size.")
        return

    esquerda, direita = st.columns(2)
    with esquerda:
        st.markdown("#### 📊 Similaridade entre chunks")
        rotulos = [f"C{i}" for i in range(1, len(chunks) + 1)]
        st.pyplot(heatmap(cosine_similarity(vetores), rotulos, "Chunks × chunks", cmap="Blues",
                          anotar=len(chunks) <= 12))
    with direita:
        st.markdown("#### 🌐 Chunks no espaço 2D (PCA)")
        coords = PCA(n_components=2, random_state=42).fit_transform(np.vstack([vetores, vetor_texto]))
        pontos = pd.DataFrame({"x": coords[:-1, 0], "y": coords[:-1, 1], "chunk": rotulos, "texto": chunks})
        estrela = pd.DataFrame({"x": [coords[-1, 0]], "y": [coords[-1, 1]], "chunk": ["texto inteiro"]})
        grafico = (
            alt.Chart(pontos).mark_circle(size=180, color="#2a9d8f", stroke="black").encode(
                x=alt.X("x", title="Componente 1"), y=alt.Y("y", title="Componente 2"), tooltip=["chunk", "texto"])
            + alt.Chart(pontos).mark_text(dx=14, dy=-8).encode(x="x", y="y", text="chunk")
            + alt.Chart(estrela).mark_point(shape="diamond", size=300, filled=True, color="#e76f51").encode(
                x="x", y="y", tooltip=["chunk"])
        )
        st.altair_chart(grafico.properties(height=380), use_container_width=True)
        st.caption("◆ = embedding do texto inteiro")

    st.markdown("#### 🔍 Buscar dentro dos chunks")
    pergunta = st.text_input("Pergunta sobre a manifestação", placeholder="ex.: faltava algum equipamento?")
    if pergunta.strip():
        scores = vetores @ gerar_embeddings(nome_modelo, (pergunta,))[0]
        for posicao, i in enumerate(np.argsort(scores)[::-1][:3], start=1):
            emoji, _ = faixa_de_score(scores[i])
            st.markdown(f"{emoji} **{posicao}º · Chunk {i + 1}** (score {scores[i]:.3f})")
            st.info(chunks[i])


# Interface principal: barra lateral com modelo e top-k e as quatro abas
def main():
    st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")
    st.title("🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs · embeddings, busca vetorial e chunking")

    st.sidebar.header("⚙️ Configurações")
    rotulo_modelo = st.sidebar.selectbox("Modelo de embedding", list(MODELOS))
    nome_modelo = MODELOS[rotulo_modelo]
    top_k = st.sidebar.slider("Top-K resultados na busca", 1, 10, 5)
    limiar_dup = st.sidebar.slider("Limiar para possível duplicata", 0.50, 0.95, 0.80, 0.01)
    metodo = st.sidebar.radio("Redução para 2D", ["PCA", "t-SNE"], horizontal=True)
    st.sidebar.caption(f"Modelo em uso: `{nome_modelo}`")

    base = carregar_base()
    try:
        emb_base = gerar_embeddings(nome_modelo, tuple(base["texto"]))
    except Exception as erro:
        st.error(f"Não foi possível carregar o modelo `{nome_modelo}`. Verifique a conexão para o download.\n\n{erro}")
        st.stop()
    st.sidebar.success(f"✅ {len(base)} manifestações indexadas ({emb_base.shape[1]} dimensões)")

    abas = st.tabs(["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"])
    with abas[0]:
        aba_busca(base, nome_modelo, emb_base, top_k, limiar_dup)
    with abas[1]:
        aba_base(base, emb_base, limiar_dup)
    with abas[2]:
        aba_espaco(base, nome_modelo, emb_base, metodo)
    with abas[3]:
        aba_chunking(base, nome_modelo)


if __name__ == "__main__":
    main()
