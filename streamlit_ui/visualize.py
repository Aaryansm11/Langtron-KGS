# visualize.py
import streamlit as st, requests, tempfile
from pyvis.network import Network

API = "http://localhost:8000"

st.set_page_config(page_title="Medical KG Explorer", layout="wide")

uploaded = st.file_uploader("Upload clinical document", ["pdf", "docx", "txt", "rtf"])
if uploaded:
    files = {"file": uploaded}
    r = requests.post(f"{API}/upload", files=files)
    if r.status_code == 200:
        doc_id = r.json()["doc_id"]
        st.success("Processed! doc_id=" + doc_id)
        graph = requests.get(f"{API}/graph/{doc_id}").json()
        if graph.get("graph"):
            net = Network(height="600px", width="100%")
            for item in graph["graph"]:
                s, t = item["e1"], item["e2"]
                net.add_node(s["name"], label=s["name"], group=s["type"])
                net.add_node(t["name"], label=t["name"], group=t["type"])
                net.add_edge(s["name"], t["name"], title=item["r"]["type"])
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                net.save_graph(f.name)
                st.components.v1.html(open(f.name).read(), height=600)