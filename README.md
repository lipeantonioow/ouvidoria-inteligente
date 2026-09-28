# Ouvidoria Inteligente

Triagem semântica de manifestações cidadãs com representações vetoriais, detecção de duplicatas e chunking, desenvolvida para o Desafio Ouvidoria Inteligente (UNIPÊ).

## Entregas

- `análise_comparativa.ipynb`: BoW × TF-IDF × embeddings (Entrega 1)
- `deteccao_duplicatas.ipynb`: detecção de duplicatas e escolha do limiar (Entrega 2)
- `chunking_manifestacoes.ipynb`: chunking das manifestações longas (Entrega 3)
- `app_ouvidoria.py`: aplicação Streamlit com busca semântica, base, espaço vetorial e chunking (Entrega 4)
- `RELATORIO.pdf`: relatório técnico
- `manifestacoes.json`: base com 40 manifestações

## Como executar

```
pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

Autor: Felipe Antônio Luna Medeiros
