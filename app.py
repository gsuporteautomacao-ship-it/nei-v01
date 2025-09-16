import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from jsonpath_ng import parse as jsonpath_parse
    HAS_JSONPATH = True
except Exception:
    HAS_JSONPATH = False


st.set_page_config(page_title="JSON Q&A Bot", page_icon="🧠", layout="wide")

# ---- Helpers ----

def flatten_json(obj: Any, path: str = "") -> List[Dict[str, Any]]:
    """
    Flatten nested JSON into rows with (path, type, value_str, value_raw).
    """
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            rows.extend(flatten_json(v, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            rows.extend(flatten_json(v, new_path))
    else:
        # leaf
        vtype = type(obj).__name__
        try:
            vstr = json.dumps(obj, ensure_ascii=False)
        except Exception:
            vstr = str(obj)
        rows.append({"path": path, "type": vtype, "value_str": vstr, "value_raw": obj})
    return rows


def build_index(df: pd.DataFrame):
    """
    Build a TF-IDF index from path and value_str columns.
    """
    if df.empty:
        return None, None
    corpus = (df["path"].astype(str) + " :: " + df["value_str"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def search(df: pd.DataFrame, vectorizer: TfidfVectorizer, matrix, query: str, top_k: int = 5):
    """
    Return top_k matching rows with scores.
    """
    if df.empty or vectorizer is None or matrix is None:
        return pd.DataFrame(columns=["score", "path", "type", "value_str"])
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    idx = np.argsort(-sims)[:top_k]
    results = df.iloc[idx].copy()
    results.insert(0, "score", sims[idx])
    return results


def synthesize_answer(question: str, hits: pd.DataFrame) -> str:
    """
    Heuristic answer synthesis without external LLMs.
    """
    if hits.empty:
        return "Não encontrei nada diretamente relacionado na base JSON. Tente reformular a pergunta ou use JSONPath."
    lines = []
    lines.append("Aqui está o que encontrei relacionado à sua pergunta:")
    for _, row in hits.iterrows():
        score = row["score"]
        path = row["path"]
        vtype = row["type"]
        val = row["value_str"]
        # format small values inline; larger values in code block
        if len(val) <= 80 and "\n" not in val:
            lines.append(f"- **{path}** ({vtype}): {val}")
        else:
            lines.append(f"- **{path}** ({vtype}):")
            lines.append(f"```\n{val}\n```")
    lines.append("Se precisar, refine a pergunta citando o caminho (path) desejado.")
    return "\n".join(lines)


def run_jsonpath_query(data_obj: Any, expr: str) -> List[Any]:
    if not HAS_JSONPATH:
        return ["jsonpath-ng não está instalado. Ative-o no requirements.txt."]
    try:
        jp = jsonpath_parse(expr)
        return [match.value for match in jp.find(data_obj)]
    except Exception as e:
        return [f"Erro JSONPath: {e}"]


# ---- Sidebar ----

with st.sidebar:
    st.markdown("## Configuração")
    st.write("Carregue um arquivo JSON ou cole o conteúdo abaixo.")
    uploaded = st.file_uploader("Arquivo .json", type=["json"])
    pasted = st.text_area("Ou cole JSON aqui", height=200, placeholder='{"empresa": {"nome": "Neoyama", "itens": [{"sku": "A6-400W", "preco": 1234.56}]}}')
    example_btn = st.button("Carregar exemplo")

# ---- Load data ----

DEFAULT_JSON = {
    "empresa": {
        "nome": "Exemplo S.A.",
        "departamentos": ["Vendas", "Compras", "TI"],
        "itens": [
            {"sku": "ABC-123", "descricao": "Motor NEMA 23", "preco": 299.9, "estoque": 12},
            {"sku": "A6-400W", "descricao": "Servo Panasonic 400W", "preco": 3899.0, "estoque": 3},
        ],
        "endereco": {"cidade": "Curitiba", "uf": "PR", "pais": "Brasil"},
        "ativo": True
    }
}

data_text = None
if example_btn:
    data_text = json.dumps(DEFAULT_JSON, ensure_ascii=False, indent=2)
elif uploaded is not None:
    data_text = uploaded.read().decode("utf-8", errors="ignore")
elif pasted.strip():
    data_text = pasted

if not data_text:
    st.info("Use a barra lateral para enviar um arquivo JSON, colar conteúdo ou carregar o exemplo.")
    st.stop()

# Validate / parse JSON
try:
    data_obj = json.loads(data_text)
except Exception as e:
    st.error(f"JSON inválido: {e}")
    st.stop()

# Flatten
rows = flatten_json(data_obj)
df = pd.DataFrame(rows, columns=["path", "type", "value_str", "value_raw"])
st.success(f"JSON carregado! {len(df)} valores de folha detectados.")
with st.expander("Ver tabela flatten (path ➜ valor)"):
    st.dataframe(df[["path", "type", "value_str"]], use_container_width=True, hide_index=True)

# Build index
vectorizer, matrix = build_index(df)

# ---- Chat UI ----

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 Chatbot de JSON (local, sem API)")

st.caption("Faça perguntas em linguagem natural ou rode uma consulta JSONPath (ex: `$..itens[?(@.preco > 500)]`).")

# JSONPath box
with st.expander("🔎 JSONPath (opcional)"):
    if HAS_JSONPATH:
        jp_expr = st.text_input("Expressão JSONPath", value="")
        if jp_expr:
            jp_res = run_jsonpath_query(data_obj, jp_expr)
            st.write("Resultado JSONPath:")
            st.json(jp_res)
    else:
        st.info("Instale `jsonpath-ng` (já no requirements.txt).")

# Chat input
user_q = st.chat_input("Digite sua pergunta sobre o JSON…")
if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    # retrieve
    hits = search(df, vectorizer, matrix, user_q, top_k=6)
    answer = synthesize_answer(user_q, hits)
    st.session_state.history.append({"role": "assistant", "content": answer, "hits": hits.to_dict(orient="records")})

# Render conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            # Show top matches nicely
            if "hits" in msg and msg["hits"]:
                with st.expander("Ver trechos relevantes"):
                    hdf = pd.DataFrame(msg["hits"])
                    # Keep only key columns for display
                    show_cols = ["score", "path", "type", "value_str"]
                    cols = [c for c in show_cols if c in hdf.columns]
                    st.dataframe(hdf[cols], use_container_width=True, hide_index=True)
        else:
            st.markdown(msg["content"])

st.markdown("---")
st.caption("Dica: para respostas mais precisas, tente perguntas específicas como “qual é o preço do SKU A6-400W?” ou use JSONPath.")