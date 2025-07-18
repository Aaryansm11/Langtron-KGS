# kg_builder.py
from py2neo import Graph, Node, Relationship, NodeMatcher
import uuid, logging
from config import NEO4J_URI, NEO4J_USER, NEO4J_PWD

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        self.matcher = NodeMatcher(self.graph)
        self._create_constraints()

    def _create_constraints(self):
        try:
            self.graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Entity) ASSERT e.uid IS UNIQUE")
        except Exception as e:
            logger.warning("Constraint creation failed: %s", e)

    def process_document(self, file_path):
        doc_id = str(uuid.uuid4())
        from app.ner import NERPipeline
        from app.relation_extractor import RelationExtractor
        ner = NERPipeline()
        re_ = RelationExtractor()

        entities = ner.extract_entities(file_path)
        relations = re_.extract_relations(entities)
        self._create_graph(doc_id, entities, relations)
        return doc_id

    def _create_graph(self, doc_id, entities, relations):
        tx = self.graph.begin()
        try:
            nodes = {}
            for ent in entities:
                node = Node("Entity",
                            uid=f"{doc_id}_{ent['id']}",
                            name=ent["text"],
                            type=ent["type"],
                            doc_id=doc_id)
                tx.create(node)
                nodes[ent["id"]] = node

            for rel in relations:
                src = nodes[rel["source_id"]]
                tgt = nodes[rel["target_id"]]
                rel_edge = Relationship(src, rel["type"], tgt)
                tx.create(rel_edge)
            tx.commit()
        except Exception as e:
            tx.rollback()
            logger.error("Neo4j write error: %s", e)
            raise