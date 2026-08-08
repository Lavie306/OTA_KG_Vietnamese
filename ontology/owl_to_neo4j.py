"""
ontology/owl_to_neo4j.py
Load TẤT CẢ OWL files → merge ontology → import vào Neo4j KG.
Chạy 1 lần trước khi train: python ontology/owl_to_neo4j.py

Tạo trong Neo4j:
  Nodes  : (:Word {name, sentiment_score, ontology_class})
  Nodes  : (:OntologyClass {name})
  Rels   : [:IS_POSITIVE], [:IS_NEGATIVE], [:IS_NEGATOR], [:IS_INTENSIFIER]
  Rels   : [:IS_SUBCLASS_OF] giữa các OntologyClass
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OWL_FILES, OWL_FILE, POSITIVE_FILE, NEGATIVE_FILE, NEGATION_FILE,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)


# ── Parse word lists từ positive.txt / negative.txt ──────────────────────────

def load_word_list(file_path: Path) -> list[str]:
    """
    Đọc file positive.txt / negative.txt.
    Format: "word1","word2","word3",... (comma-separated, có dấu nháy kép)
    """
    text = file_path.read_text(encoding="utf-8")
    words = [w.strip().strip('"') for w in text.split(",")]
    words = [w for w in words if w]
    return words


def load_negation_words(file_path: Path) -> list[str]:
    """Đọc negation_words.csv — mỗi dòng là một từ."""
    words = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words


# ── Parse OWL bằng owlready2 ─────────────────────────────────────────────────

def parse_owl_file(owl_path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """
    Parse 1 OWL file bằng owlready2.
    Returns:
        instances: dict {class_name: [instance_names]}
        subclass_rels: list[(child_class, parent_class)]
    """
    try:
        import owlready2
        onto = owlready2.get_ontology(str(owl_path.resolve())).load()
        instances = {}
        subclass_rels = []

        for cls in onto.classes():
            class_name = cls.name
            # Collect instances
            inds = [ind.name for ind in cls.instances()]
            if inds:
                instances[class_name] = inds

            # Collect SubClassOf relationships
            for parent in cls.is_a:
                if hasattr(parent, "name") and parent.name != "Thing":
                    subclass_rels.append((class_name, parent.name))

        print(f"  [OWL] {owl_path.name}: {len(instances)} classes with instances, "
              f"{len(subclass_rels)} subclass rels")
        return instances, subclass_rels

    except Exception as e:
        print(f"  [OWL] Không parse được {owl_path.name}: {e}")
        return {}, []


def parse_all_owl_files(owl_files: list[Path]) -> tuple[dict, set[tuple[str, str]], set[str]]:
    """
    Parse TẤT CẢ OWL files, merge kết quả.
    Returns:
        all_instances: dict {class_name: set(instance_names)}  — merged
        all_subclass_rels: set of (child, parent) tuples
        all_classes: set of all class names
    """
    all_instances: dict[str, set] = {}
    all_subclass_rels: set[tuple[str, str]] = set()
    all_classes: set[str] = set()

    # Classes được định nghĩa qua SubClassOf trong OWL files (manual fallback)
    # Nếu owlready2 không parse được, dùng fallback mapping từ phân tích thủ công
    FALLBACK_SUBCLASS = {
        # ver3/ver5 classes (Chatbot_ontology)
        "Noun":        "Word",
        "Verb":        "Word",
        "Adjective":   "Word",
        "Adverb":      "Word",
        "Conjunction": "Sentence",
        "Coordinating": "Conjunction",
        "Subordinating": "Conjunction",
        "Negative":    "Sentence",
        "Positive":    "Sentence",
        "Negative_Adverbs": "Sentence",
        "Prefix":      "Sentence",
        "Special_sentence": "Sentence",
        # ver12 classes
        "Emotion":     "Sentence",
        "SentimentAnalysis": "Negative",  # SubClass cả Negative lẫn Positive
        "IntensifiersAndNegators": "Word",
        "EmotionWords": "Word",
        # ver12: thêm Words layer
        "Words":       "Sentence",
        "subject":     "Sentence",
        "predicate":   "Sentence",
    }

    for owl_path in owl_files:
        if not owl_path.exists():
            print(f"  [OWL] File không tồn tại, bỏ qua: {owl_path.name}")
            continue

        instances, subclass_rels = parse_owl_file(owl_path)

        # Merge instances
        for cls_name, inds in instances.items():
            if cls_name not in all_instances:
                all_instances[cls_name] = set()
            all_instances[cls_name].update(inds)

        # Merge subclass rels
        all_subclass_rels.update(subclass_rels)

    # Thêm fallback subclass nếu không có từ parse
    if not all_subclass_rels:
        print("  [OWL] Dùng fallback SubClassOf mapping (owlready2 không parse được)")
        all_subclass_rels.update(FALLBACK_SUBCLASS.items())

    # Collect tất cả class names
    all_classes.update(all_instances.keys())
    for child, parent in all_subclass_rels:
        all_classes.add(child)
        all_classes.add(parent)

    # Thêm root classes
    ROOT_CLASSES = {
        "Sentence", "Word", "Conjunction", "Negative", "Positive",
        "Negative_Adverbs", "Prefix", "Special_sentence",
        "Coordinating", "Subordinating", "Noun", "Verb", "Adjective", "Adverb",
        "Emotion", "SentimentAnalysis", "IntensifiersAndNegators", "EmotionWords",
        "Words", "subject", "predicate",
    }
    all_classes.update(ROOT_CLASSES)
    if not all_subclass_rels:
        all_subclass_rels.update(FALLBACK_SUBCLASS.items())

    print(f"\n[OWL] Tổng hợp: {len(all_classes)} classes, "
          f"{sum(len(v) for v in all_instances.values())} instances, "
          f"{len(all_subclass_rels)} subclass rels")

    return all_instances, all_subclass_rels, all_classes


# ── Import vào Neo4j ──────────────────────────────────────────────────────────

def import_to_neo4j(
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
) -> None:
    """
    Import toàn bộ ontology (tất cả OWL files) + word lists vào Neo4j.

    Nodes được tạo:
        (:Word {name, sentiment_score, ontology_class})
        (:OntologyClass {name})
    Relationships:
        (:Word)-[:IS_POSITIVE]->(:SentimentClass {name:"Positive"})
        (:Word)-[:IS_NEGATIVE {weight}]->(:SentimentClass {name:"Negative"})
        (:Word)-[:IS_NEGATOR]->(:SentimentClass {name:"Negator"})
        (:OntologyClass)-[:IS_SUBCLASS_OF]->(:OntologyClass)
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[Neo4j] Chưa cài neo4j driver: pip install neo4j")
        return

    print(f"[Neo4j] Đang kết nối tới Neo4j Aura tại {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("[Neo4j] Kết nối thành công!")
    except Exception as e:
        print(f"\n[Neo4j] Không thể kết nối tới Neo4j Aura tại {uri}: {e}")
        print("[Neo4j] BỎ QUA việc đẩy dữ liệu lên Neo4j. Bạn có thể xây dựng cache offline bằng: python precompute_kg.py --offline")
        return

    # Load dữ liệu
    positive_words = load_word_list(POSITIVE_FILE)
    negative_words = load_word_list(NEGATIVE_FILE)
    negation_words = load_negation_words(NEGATION_FILE)

    # Parse TẤT CẢ OWL files
    print("\n[OWL] Đang parse tất cả OWL files...")
    all_instances, all_subclass_rels, all_classes = parse_all_owl_files(OWL_FILES)

    print(f"\n[Import] positive={len(positive_words)}, negative={len(negative_words)}, "
          f"negation={len(negation_words)}")

    with driver.session(database=user) as session:
        # Xóa data cũ
        print("[Import] Clearing old data...")
        session.run("MATCH (n:Word) DETACH DELETE n")
        session.run("MATCH (n:SentimentClass) DETACH DELETE n")
        session.run("MATCH (n:OntologyClass) DETACH DELETE n")

        # Tạo SentimentClass nodes
        for cls in ["Positive", "Negative", "Negator", "Intensifier"]:
            session.run("MERGE (:SentimentClass {name: $name})", name=cls)

        # Tạo OntologyClass nodes + IS_SUBCLASS_OF rels
        print(f"[Import] Importing {len(all_classes)} OntologyClass nodes...")
        for cls_name in all_classes:
            session.run(
                "MERGE (:OntologyClass {name: $name})",
                name=cls_name,
            )
        for child, parent in all_subclass_rels:
            session.run(
                """
                MATCH (c:OntologyClass {name: $child})
                MATCH (p:OntologyClass {name: $parent})
                MERGE (c)-[:IS_SUBCLASS_OF]->(p)
                """,
                child=child, parent=parent,
            )
        print(f"[Import] {len(all_subclass_rels)} IS_SUBCLASS_OF rels created")

        # Import positive words
        print("[Import] Importing positive words...")
        for i in range(0, len(positive_words), 500):
            batch = positive_words[i:i+500]
            session.run(
                """
                UNWIND $words AS w
                MERGE (word:Word {name: w})
                SET word.sentiment_score = 1.0,
                    word.ontology_class  = 'Positive'
                WITH word
                MATCH (cls:SentimentClass {name: 'Positive'})
                MERGE (word)-[:IS_POSITIVE]->(cls)
                """,
                words=batch,
            )

        # Import negative words
        print("[Import] Importing negative words...")
        for i in range(0, len(negative_words), 500):
            batch = negative_words[i:i+500]
            session.run(
                """
                UNWIND $words AS w
                MERGE (word:Word {name: w})
                SET word.sentiment_score = -1.0,
                    word.ontology_class  = 'Negative'
                WITH word
                MATCH (cls:SentimentClass {name: 'Negative'})
                MERGE (word)-[:IS_NEGATIVE {weight: -1}]->(cls)
                """,
                words=batch,
            )

        # Import negation words
        print("[Import] Importing negation words...")
        session.run(
            """
            UNWIND $words AS w
            MERGE (word:Word {name: w})
            SET word.sentiment_score = 0.0,
                word.ontology_class  = 'Negator'
            WITH word
            MATCH (cls:SentimentClass {name: 'Negator'})
            MERGE (word)-[:IS_NEGATOR]->(cls)
            """,
            words=negation_words,
        )

        # Import OWL instances (words từ ontology)
        print(f"[Import] Importing OWL instances ({len(all_instances)} classes)...")
        for class_name, instances in all_instances.items():
            instance_list = list(instances)
            for i in range(0, len(instance_list), 500):
                batch = instance_list[i:i+500]
                session.run(
                    """
                    UNWIND $words AS w
                    MERGE (word:Word {name: w})
                    SET word.ontology_class = $cls
                    WITH word
                    MATCH (oc:OntologyClass {name: $cls})
                    MERGE (word)-[:BELONGS_TO]->(oc)
                    """,
                    words=batch,
                    cls=class_name,
                )

        # Tạo index để query nhanh
        session.run("CREATE INDEX word_name IF NOT EXISTS FOR (w:Word) ON (w.name)")
        session.run("CREATE INDEX onto_class_name IF NOT EXISTS FOR (c:OntologyClass) ON (c.name)")

        # Đếm kết quả
        word_count  = session.run("MATCH (n:Word) RETURN count(n) AS c").single()["c"]
        class_count = session.run("MATCH (n:OntologyClass) RETURN count(n) AS c").single()["c"]
        print(f"\n[Import] Done!")
        print(f"  Word nodes        : {word_count}")
        print(f"  OntologyClass nodes: {class_count}")

    driver.close()


# ── Cypher query để verify ────────────────────────────────────────────────────

VERIFY_QUERIES = """
-- Kiểm tra sau khi import:
MATCH (w:Word) RETURN count(w);                              // Tổng số từ
MATCH (w:Word {ontology_class:'Positive'}) RETURN count(w); // Positive
MATCH (w:Word {ontology_class:'Negative'}) RETURN count(w); // Negative
MATCH (w:Word {ontology_class:'Negator'})  RETURN count(w); // Negation
MATCH (c:OntologyClass) RETURN c.name ORDER BY c.name;      // Tất cả classes
MATCH (c)-[:IS_SUBCLASS_OF]->(p) RETURN c.name, p.name;    // Hierarchy
MATCH (w:Word {name:'not'}) RETURN w;                        // Test từ cụ thể
MATCH (w:Word {name:'amazing'}) RETURN w;
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Import ALL OWL Ontologies -> Neo4j Knowledge Graph")
    print("=" * 60)
    import_to_neo4j()
    print("\nVerify bằng Neo4j Browser:")
    print(VERIFY_QUERIES)
