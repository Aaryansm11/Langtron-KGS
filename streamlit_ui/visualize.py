# visualize.py
import streamlit as st
import requests
import tempfile
import time
from pyvis.network import Network

API = "http://localhost:8000"

st.set_page_config(page_title="Medical KG Explorer", layout="wide")

st.title("Medical Knowledge Graph Explorer")
st.markdown("Upload a clinical document to extract entities, relations, and visualize the knowledge graph.")

uploaded = st.file_uploader("Upload clinical document", ["pdf", "docx", "txt", "rtf"])

if uploaded:
    with st.spinner("Uploading document..."):
        files = {"file": uploaded}
        r = requests.post(f"{API}/upload", files=files)
        
        if r.status_code == 200:
            response_data = r.json()
            task_id = response_data["task_id"]  # Use task_id instead of doc_id
            st.success(f"Document uploaded successfully! Task ID: {task_id}")
            
            # Poll for processing status
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            while True:
                status_response = requests.get(f"{API}/status/{task_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    # Update progress
                    progress_bar.progress(status_data.get("progress", 0))
                    status_placeholder.info(f"Status: {status_data['status']} - {status_data['message']}")
                    
                    if status_data["status"] == "completed":
                        st.success("Processing completed!")
                        
                        # Get the result from the status response
                        result = status_data.get("result", {})
                        
                        if result:
                            # Display statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Entities", result["metadata"]["total_entities"])
                            with col2:
                                st.metric("Relations", result["metadata"]["total_relations"])
                            with col3:
                                st.metric("Graph Nodes", result["metadata"]["graph_nodes"])
                            with col4:
                                st.metric("Graph Edges", result["metadata"]["graph_edges"])
                            
                            # Display entities
                            if result.get("entities"):
                                st.subheader("Extracted Entities")
                                entities_df = []
                                for entity in result["entities"]:
                                    entities_df.append({
                                        "Text": entity.get("text", ""),
                                        "Label": entity.get("label", ""),
                                        "Confidence": f"{entity.get('confidence', 0):.2f}",
                                        "Start": entity.get("start", ""),
                                        "End": entity.get("end", "")
                                    })
                                st.dataframe(entities_df)
                            
                            # Display relations
                            if result.get("relations"):
                                st.subheader("Extracted Relations")
                                relations_df = []
                                for relation in result["relations"]:
                                    relations_df.append({
                                        "Subject": relation.get("subject", ""),
                                        "Relation": relation.get("relation", ""),
                                        "Object": relation.get("object", ""),
                                        "Confidence": f"{relation.get('confidence', 0):.2f}"
                                    })
                                st.dataframe(relations_df)
                            
                            # Visualize knowledge graph
                            if result.get("graph") and result["graph"].get("nodes"):
                                st.subheader("Knowledge Graph Visualization")
                                
                                # Create network visualization
                                net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
                                net.set_options("""
                                var options = {
                                  "physics": {
                                    "enabled": true,
                                    "stabilization": {"iterations": 100}
                                  }
                                }
                                """)
                                
                                # Add nodes
                                for node in result["graph"]["nodes"]:
                                    net.add_node(
                                        node["id"],
                                        label=node["label"],
                                        group=node.get("type", "default"),
                                        size=node.get("size", 10),
                                        title=f"Type: {node.get('type', 'Unknown')}\nConfidence: {node.get('confidence', 0):.2f}"
                                    )
                                
                                # Add edges
                                for edge in result["graph"]["edges"]:
                                    net.add_edge(
                                        edge["source"],
                                        edge["target"],
                                        label=edge.get("relation", ""),
                                        title=f"Relation: {edge.get('relation', 'Unknown')}\nConfidence: {edge.get('confidence', 0):.2f}",
                                        color=edge.get("color", "#848484")
                                    )
                                
                                # Save and display graph
                                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                                    net.save_graph(f.name)
                                    with open(f.name, 'r') as file:
                                        graph_html = file.read()
                                    st.components.v1.html(graph_html, height=600)
                                
                                # Download options
                                st.subheader("Download Options")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("Download Graph as JSON"):
                                        st.download_button(
                                            label="Download JSON",
                                            data=str(result["graph"]),
                                            file_name="knowledge_graph.json",
                                            mime="application/json"
                                        )
                                
                                with col2:
                                    # Option to get graph in other formats
                                    format_option = st.selectbox("Graph Format", ["gexf", "graphml"])
                                    if st.button(f"Download as {format_option.upper()}"):
                                        # Use the direct text extraction endpoint for different formats
                                        if uploaded.type == "text/plain":
                                            text_content = str(uploaded.getvalue(), "utf-8")
                                            graph_response = requests.get(
                                                f"{API}/graph", 
                                                params={"text": text_content, "format": format_option}
                                            )
                                            if graph_response.status_code == 200:
                                                st.download_button(
                                                    label=f"Download {format_option.upper()}",
                                                    data=graph_response.content,
                                                    file_name=f"knowledge_graph.{format_option}",
                                                    mime="application/octet-stream"
                                                )
                            else:
                                st.warning("No graph data available for visualization.")
                        else:
                            st.warning("No result data available.")
                        break
                        
                    elif status_data["status"] == "error":
                        st.error(f"Processing failed: {status_data.get('error', 'Unknown error')}")
                        break
                    
                    # Wait before next poll
                    time.sleep(2)
                else:
                    st.error("Failed to get processing status")
                    break
        else:
            st.error(f"Upload failed: {r.status_code} - {r.text}")

# Add sidebar with information
st.sidebar.header("About")
st.sidebar.info("""
This application extracts medical entities and relations from clinical documents 
and visualizes them as an interactive knowledge graph.

**Supported formats:**
- PDF
- DOCX  
- TXT
- RTF

**Features:**
- Named Entity Recognition
- Relation Extraction
- Knowledge Graph Construction
- Interactive Visualization
- Multiple Export Formats
""")

st.sidebar.header("API Status")
try:
    health_response = requests.get(f"{API}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data["status"] == "healthy":
            st.sidebar.success("API is healthy ✅")
        else:
            st.sidebar.warning("API is unhealthy ⚠️")
        
        # Show model status
        st.sidebar.text("Model Status:")
        for model, status in health_data["models"].items():
            status_icon = "✅" if status else "❌"
            st.sidebar.text(f"{model}: {status_icon}")
    else:
        st.sidebar.error("API is not responding ❌")
except:
    st.sidebar.error("Cannot connect to API ❌")