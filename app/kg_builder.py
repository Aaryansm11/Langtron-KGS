#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder with comprehensive functionality
"""

import uuid
import logging
import time
import json
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from py2neo.bulk import create_nodes, create_relationships
import spacy
from sentence_transformers import SentenceTransformer

from utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Enhanced node representation"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 0.0
    frequency: int = 1
    centrality: float = 0.0


@dataclass
class GraphRelation:
    """Enhanced relationship representation"""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 0.0
    frequency: int = 1
    weight: float = 1.0


class KnowledgeGraphBuilder:
    """Enhanced Knowledge Graph Builder with Neo4j integration"""
    
    def __init__(self):
        self.graph = None
        self.matcher = None
        self.rel_matcher = None
        self.similarity_model = None
        self._initialize_connections()
        self._initialize_models()
        self._create_constraints()
        
        # Graph statistics
        self.stats = {
            'nodes_created': 0,
            'relationships_created': 0,
            'duplicates_merged': 0,
            'processing_time': 0.0
        }
    
    def _initialize_connections(self):
        """Initialize Neo4j connections with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.graph = Graph(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    name=settings.NEO4J_DATABASE
                )
                self.matcher = NodeMatcher(self.graph)
                self.rel_matcher = RelationshipMatcher(self.graph)
                
                # Test connection
                self.graph.run("RETURN 1")
                logger.info("Successfully connected to Neo4j")
                return
                
            except Exception as e:
                logger.warning(f"Neo4j connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Could not connect to Neo4j after multiple attempts")
                    raise
                time.sleep(2 ** attempt)
    
    def _initialize_models(self):
        """Initialize similarity models"""
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Similarity model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load similarity model: {e}")
            self.similarity_model = None
    
    def _create_constraints(self):
        """Create Neo4j constraints and indices"""
        constraints = [
            "CREATE CONSTRAINT entity_uid IF NOT EXISTS ON (e:Entity) ASSERT e.uid IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS ON (e:Entity) ASSERT e.name IS NOT NULL",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.doc_id)",
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases]"
        ]
        
        for constraint in constraints:
            try:
                self.graph.run(constraint)
                logger.debug(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint creation failed: {e}")
    
    def build_knowledge_graph(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """Build knowledge graph from entities and relations"""
        start_time = time.time()
        
        try:
            # Create graph nodes
            nodes = self._create_nodes(entities)
            
            # Create graph relationships
            edges = self._create_relationships(relations, nodes)
            
            # Build NetworkX graph for analysis
            nx_graph = self._build_networkx_graph(nodes, edges)
            
            # Calculate graph metrics
            enhanced_nodes = self._calculate_node_metrics(nx_graph, nodes)
            enhanced_edges = self._calculate_edge_metrics(nx_graph, edges)
            
            # Update statistics
            self.stats['processing_time'] = time.time() - start_time
            self.stats['nodes_created'] = len(nodes)
            self.stats['relationships_created'] = len(edges)
            
            graph_data = {
                'nodes': [self._node_to_dict(node) for node in enhanced_nodes],
                'edges': [self._edge_to_dict(edge) for edge in enhanced_edges],
                'statistics': self._calculate_graph_statistics(nx_graph),
                'metadata': {
                    'created_at': time.time(),
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'processing_time': self.stats['processing_time']
                }
            }
            
            logger.info(f"Knowledge graph built with {len(nodes)} nodes and {len(edges)} edges")
            return graph_data
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise
    
    def _create_nodes(self, entities: List[Dict]) -> List[GraphNode]:
        """Create graph nodes from entities"""
        nodes = []
        entity_map = {}
        
        for entity in entities:
            # Create unique ID
            entity_id = entity.get('id', str(uuid.uuid4()))
            
            # Handle duplicate entities
            entity_key = (entity['text'].lower(), entity['type'])
            if entity_key in entity_map:
                existing_node = entity_map[entity_key]
                existing_node.frequency += 1
                existing_node.confidence = max(existing_node.confidence, entity.get('confidence', 0.0))
                continue
            
            # Create new node
            node = GraphNode(
                id=entity_id,
                name=entity['text'],
                type=entity['type'],
                properties={
                    'start_pos': entity.get('start', 0),
                    'end_pos': entity.get('end', 0),
                    'normalized': entity.get('normalized', ''),
                    'aliases': entity.get('aliases', []),
                    'description': entity.get('description', ''),
                    'wikidata_id': entity.get('wikidata_id', ''),
                    'created_at': time.time()
                },
                confidence=entity.get('confidence', 0.0),
                frequency=1
            )
            
            nodes.append(node)
            entity_map[entity_key] = node
        
        return nodes
    
    def _create_relationships(self, relations: List[Dict], nodes: List[GraphNode]) -> List[GraphRelation]:
        """Create graph relationships from relations"""
        edges = []
        node_map = {node.id: node for node in nodes}
        relation_map = {}
        
        for relation in relations:
            source_id = relation.get('source_id', '')
            target_id = relation.get('target_id', '')
            
            # Skip if nodes don't exist
            if source_id not in node_map or target_id not in node_map:
                continue
            
            # Handle duplicate relations
            relation_key = (source_id, target_id, relation['type'])
            if relation_key in relation_map:
                existing_rel = relation_map[relation_key]
                existing_rel.frequency += 1
                existing_rel.confidence = max(existing_rel.confidence, relation.get('confidence', 0.0))
                continue
            
            # Create new relationship
            edge = GraphRelation(
                id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                type=relation['type'],
                properties={
                    'context': relation.get('context', ''),
                    'sentence': relation.get('sentence', ''),
                    'created_at': time.time()
                },
                confidence=relation.get('confidence', 0.0),
                frequency=1,
                weight=relation.get('weight', 1.0)
            )
            
            edges.append(edge)
            relation_map[relation_key] = edge
        
        return edges
    
    def _build_networkx_graph(self, nodes: List[GraphNode], edges: List[GraphRelation]) -> nx.Graph:
        """Build NetworkX graph for analysis"""
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id, **{
                'name': node.name,
                'type': node.type,
                'confidence': node.confidence,
                'frequency': node.frequency
            })
        
        # Add edges
        for edge in edges:
            G.add_edge(edge.source_id, edge.target_id, **{
                'type': edge.type,
                'confidence': edge.confidence,
                'frequency': edge.frequency,
                'weight': edge.weight
            })
        
        return G
    
    def _calculate_node_metrics(self, graph: nx.Graph, nodes: List[GraphNode]) -> List[GraphNode]:
        """Calculate node centrality metrics"""
        try:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            closeness_centrality = nx.closeness_centrality(graph)
            
            # Update nodes with centrality scores
            for node in nodes:
                node.centrality = degree_centrality.get(node.id, 0.0)
                node.properties.update({
                    'degree_centrality': degree_centrality.get(node.id, 0.0),
                    'betweenness_centrality': betweenness_centrality.get(node.id, 0.0),
                    'closeness_centrality': closeness_centrality.get(node.id, 0.0)
                })
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error calculating node metrics: {e}")
            return nodes
    
    def _calculate_edge_metrics(self, graph: nx.Graph, edges: List[GraphRelation]) -> List[GraphRelation]:
        """Calculate edge metrics"""
        try:
            # Calculate edge betweenness
            edge_betweenness = nx.edge_betweenness_centrality(graph)
            
            # Update edges with metrics
            for edge in edges:
                edge_key = (edge.source_id, edge.target_id)
                edge.properties.update({
                    'edge_betweenness': edge_betweenness.get(edge_key, 0.0)
                })
            
            return edges
            
        except Exception as e:
            logger.error(f"Error calculating edge metrics: {e}")
            return edges
    
    def _calculate_graph_statistics(self, graph: nx.Graph) -> Dict:
        """Calculate comprehensive graph statistics"""
        try:
            return {
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'density': nx.density(graph),
                'average_clustering': nx.average_clustering(graph),
                'connected_components': nx.number_connected_components(graph),
                'diameter': nx.diameter(graph) if nx.is_connected(graph) else 0,
                'average_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0,
                'node_types': self._get_node_type_distribution(graph),
                'edge_types': self._get_edge_type_distribution(graph)
            }
        except Exception as e:
            logger.error(f"Error calculating graph statistics: {e}")
            return {}
    
    def _get_node_type_distribution(self, graph: nx.Graph) -> Dict:
        """Get distribution of node types"""
        type_counts = defaultdict(int)
        for node_id in graph.nodes():
            node_type = graph.nodes[node_id].get('type', 'UNKNOWN')
            type_counts[node_type] += 1
        return dict(type_counts)
    
    def _get_edge_type_distribution(self, graph: nx.Graph) -> Dict:
        """Get distribution of edge types"""
        type_counts = defaultdict(int)
        for edge in graph.edges(data=True):
            edge_type = edge[2].get('type', 'UNKNOWN')
            type_counts[edge_type] += 1
        return dict(type_counts)
    
    def enhance_graph(self, graph_data: Dict) -> Dict:
        """Enhance graph with additional features"""
        try:
            # Add entity linking
            if self.similarity_model:
                graph_data = self._add_entity_linking(graph_data)
            
            # Add community detection
            graph_data = self._add_community_detection(graph_data)
            
            # Add node importance ranking
            graph_data = self._add_importance_ranking(graph_data)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error enhancing graph: {e}")
            return graph_data
    
    def _add_entity_linking(self, graph_data: Dict) -> Dict:
        """Add entity linking using similarity"""
        try:
            nodes = graph_data['nodes']
            node_texts = [node['name'] for node in nodes]
            
            if len(node_texts) > 1:
                # Calculate similarities
                embeddings = self.similarity_model.encode(node_texts)
                similarities = self.similarity_model.similarity(embeddings, embeddings)
                
                # Find similar entities
                similar_pairs = []
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        if similarities[i][j] > settings.SIMILARITY_THRESHOLD:
                            similar_pairs.append({
                                'node1': nodes[i]['id'],
                                'node2': nodes[j]['id'],
                                'similarity': float(similarities[i][j])
                            })
                
                graph_data['similar_entities'] = similar_pairs
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error in entity linking: {e}")
            return graph_data
    
    def _add_community_detection(self, graph_data: Dict) -> Dict:
        """Add community detection results"""
        try:
            # Build NetworkX graph
            G = nx.Graph()
            
            for node in graph_data['nodes']:
                G.add_node(node['id'], **node)
            
            for edge in graph_data['edges']:
                G.add_edge(edge['source_id'], edge['target_id'], **edge)
            
            # Detect communities using Louvain algorithm
            communities = nx.community.louvain_communities(G)
            
            # Add community info to nodes
            community_map = {}
            for i, community in enumerate(communities):
                for node_id in community:
                    community_map[node_id] = i
            
            for node in graph_data['nodes']:
                node['community'] = community_map.get(node['id'], -1)
            
            graph_data['communities'] = len(communities)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return graph_data
    
    def _add_importance_ranking(self, graph_data: Dict) -> Dict:
        """Add importance ranking to nodes"""
        try:
            # Sort nodes by centrality and frequency
            nodes = graph_data['nodes']
            nodes.sort(key=lambda x: (x.get('centrality', 0), x.get('frequency', 0)), reverse=True)
            
            # Add ranking
            for i, node in enumerate(nodes):
                node['importance_rank'] = i + 1
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error in importance ranking: {e}")
            return graph_data
    
    def optimize_graph(self, graph_data: Dict) -> Dict:
        """Optimize graph for performance and visualization"""
        try:
            # Filter low-confidence entities
            min_confidence = settings.NER_CONFIDENCE
            filtered_nodes = [
                node for node in graph_data['nodes']
                if node.get('confidence', 0) >= min_confidence
            ]
            
            # Keep only edges with both endpoints
            node_ids = {node['id'] for node in filtered_nodes}
            filtered_edges = [
                edge for edge in graph_data['edges']
                if edge['source_id'] in node_ids and edge['target_id'] in node_ids
            ]
            
            # Limit graph size for visualization
            max_nodes = 1000
            if len(filtered_nodes) > max_nodes:
                # Keep most important nodes
                filtered_nodes = sorted(
                    filtered_nodes,
                    key=lambda x: x.get('importance_rank', float('inf'))
                )[:max_nodes]
                
                node_ids = {node['id'] for node in filtered_nodes}
                filtered_edges = [
                    edge for edge in filtered_edges
                    if edge['source_id'] in node_ids and edge['target_id'] in node_ids
                ]
            
            return {
                **graph_data,
                'nodes': filtered_nodes,
                'edges': filtered_edges,
                'optimization_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error optimizing graph: {e}")
            return graph_data
    
    def save_to_neo4j(self, graph_data: Dict, doc_id: str) -> bool:
        """Save graph to Neo4j database"""
        try:
            tx = self.graph.begin()
            
            # Create document node
            doc_node = Node("Document", 
                          doc_id=doc_id,
                          created_at=time.time(),
                          node_count=len(graph_data['nodes']),
                          edge_count=len(graph_data['edges']))
            tx.create(doc_node)
            
            # Create entity nodes
            for node_data in graph_data['nodes']:
                entity_node = Node("Entity",
                                 uid=f"{doc_id}_{node_data['id']}",
                                 doc_id=doc_id,
                                 name=node_data['name'],
                                 type=node_data['type'],
                                 confidence=node_data.get('confidence', 0.0),
                                 frequency=node_data.get('frequency', 1),
                                 centrality=node_data.get('centrality', 0.0),
                                 **node_data.get('properties', {}))
                tx.create(entity_node)
                
                # Link to document
                doc_rel = Relationship(doc_node, "CONTAINS", entity_node)
                tx.create(doc_rel)
            
            # Create relationships
            node_map = {}
            for node in tx.graph.nodes.match("Entity", doc_id=doc_id):
                node_map[node['name']] = node
            
            for edge_data in graph_data['edges']:
                source_name = next(n['name'] for n in graph_data['nodes'] if n['id'] == edge_data['source_id'])
                target_name = next(n['name'] for n in graph_data['nodes'] if n['id'] == edge_data['target_id'])
                
                if source_name in node_map and target_name in node_map:
                    source_node = node_map[source_name]
                    target_node = node_map[target_name]
                    
                    relationship = Relationship(source_node, edge_data['type'], target_node,
                                              confidence=edge_data.get('confidence', 0.0),
                                              frequency=edge_data.get('frequency', 1),
                                              weight=edge_data.get('weight', 1.0),
                                              **edge_data.get('properties', {}))
                    tx.create(relationship)
            
            tx.commit()
            logger.info(f"Successfully saved graph to Neo4j for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to Neo4j: {e}")
            tx.rollback()
            return False
    
    def get_graph_statistics(self, graph_data: Dict) -> Dict:
        """Get comprehensive graph statistics"""
        return graph_data.get('statistics', {})
    
    def save_graph(self, graph_data: Dict, file_path: str, format: str = 'json') -> bool:
        """Save graph to file in various formats"""
        try:
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(graph_data, f, indent=2)
                    
            elif format.lower() == 'gexf':
                # Convert to NetworkX and save as GEXF
                G = nx.Graph()
                
                for node in graph_data['nodes']:
                    G.add_node(node['id'], **node)
                
                for edge in graph_data['edges']:
                    G.add_edge(edge['source_id'], edge['target_id'], **edge)
                
                nx.write_gexf(G, file_path)
                
            elif format.lower() == 'graphml':
                # Convert to NetworkX and save as GraphML
                G = nx.Graph()
                
                for node in graph_data['nodes']:
                    G.add_node(node['id'], **node)
                
                for edge in graph_data['edges']:
                    G.add_edge(edge['source_id'], edge['target_id'], **edge)
                
                nx.write_graphml(G, file_path)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return False
    
    def _node_to_dict(self, node: GraphNode) -> Dict:
        """Convert GraphNode to dictionary"""
        return {
            'id': node.id,
            'name': node.name,
            'type': node.type,
            'confidence': node.confidence,
            'frequency': node.frequency,
            'centrality': node.centrality,
            'properties': node.properties
        }
    
    def _edge_to_dict(self, edge: GraphRelation) -> Dict:
        """Convert GraphRelation to dictionary"""
        return {
            'id': edge.id,
            'source_id': edge.source_id,
            'target_id': edge.target_id,
            'type': edge.type,
            'confidence': edge.confidence,
            'frequency': edge.frequency,
            'weight': edge.weight,
            'properties': edge.properties
        }
    
    def process_document(self, file_path: str) -> str:
        """Process document and create knowledge graph"""
        doc_id = str(uuid.uuid4())
        
        try:
            # Import here to avoid circular imports
            from src.document_parser import DocumentParser
            from src.ner_model import NERModel
            from src.relation_extractor import RelationExtractor
            
            # Initialize processors
            parser = DocumentParser()
            ner_model = NERModel()
            relation_extractor = RelationExtractor()
            
            # Parse document
            parsed_doc = parser.parse_document(file_path)
            
            # Extract entities
            entities = ner_model.extract_entities(parsed_doc.text)
            
            # Extract relations
            relations = relation_extractor.extract_relations(parsed_doc.text, entities)
            
            # Build knowledge graph
            graph_data = self.build_knowledge_graph(entities, relations)
            
            # Enhance and optimize graph
            enhanced_graph = self.enhance_graph(graph_data)
            optimized_graph = self.optimize_graph(enhanced_graph)
            
            # Save to Neo4j
            self.save_to_neo4j(optimized_graph, doc_id)
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def get_document_graph(self, doc_id: str) -> Dict:
        """Retrieve graph for a specific document"""
        try:
            # Query Neo4j for document entities and relationships
            query = """
            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(e:Entity)
            OPTIONAL MATCH (e)-[r]-(other:Entity)
            WHERE other.doc_id = $doc_id
            RETURN e, r, other
            """
            
            result = self.graph.run(query, doc_id=doc_id)
            
            nodes = {}
            edges = []
            
            for record in result:
                entity = record['e']
                if entity['name'] not in nodes:
                    nodes[entity['name']] = {
                        'id': entity['uid'],
                        'name': entity['name'],
                        'type': entity['type'],
                        'confidence': entity.get('confidence', 0.0),
                        'frequency': entity.get('frequency', 1),
                        'centrality': entity.get('centrality', 0.0)
                    }
                
                if record['r'] and record['other']:
                    other_entity = record['other']
                    if other_entity['name'] not in nodes:
                        nodes[other_entity['name']] = {
                            'id': other_entity['uid'],
                            'name': other_entity['name'],
                            'type': other_entity['type'],
                            'confidence': other_entity.get('confidence', 0.0),
                            'frequency': other_entity.get('frequency', 1),
                            'centrality': other_entity.get('centrality', 0.0)
                        }
                    
                    relationship = record['r']
                    edges.append({
                        'source_id': entity['uid'],
                        'target_id': other_entity['uid'],
                        'type': relationship.type,
                        'confidence': relationship.get('confidence', 0.0),
                        'frequency': relationship.get('frequency', 1),
                        'weight': relationship.get('weight', 1.0)
                    })
            
            return {
                'nodes': list(nodes.values()),
                'edges': edges,
                'metadata': {
                    'doc_id': doc_id,
                    'node_count': len(nodes),
                    'edge_count': len(edges)
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving document graph: {e}")
            return {'nodes': [], 'edges': [], 'metadata': {}}