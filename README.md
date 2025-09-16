# JSON Q&A Bot (Streamlit)

Um app no Streamlit para **fazer perguntas sobre um arquivo JSON**, sem depender de APIs externas.

## Como usar no streamlit.io (Community Cloud)

1. Crie um novo repositório no GitHub e envie estes arquivos: `app.py`, `requirements.txt` e (opcional) `sample.json`.
2. No [Streamlit Community Cloud](https://share.streamlit.io/), conecte o repositório e escolha a branch e o arquivo `app.py`.
3. Deploy. O app abrirá pedindo para **enviar um JSON**, **colar** o conteúdo, ou **carregar o exemplo**.

## Recursos

- Upload de arquivo `.json` ou colagem do conteúdo.
- Visualização do JSON "achatado" com caminhos (`path`).
- Busca por **linguagem natural** com TF‑IDF (local, sem API).
- Suporte a **JSONPath** via `jsonpath-ng` (ex: `$..itens[?(@.preco > 500)]`).
- Interface de chat com histórico.

## Dica

Pergunte algo como:
- "Qual é o preço do SKU A6-400W?"
- "Quais departamentos existem?"
- "Mostre os itens com estoque maior que 5" (use JSONPath para filtros numéricos).

## Desenvolvimento local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Licença

MIT