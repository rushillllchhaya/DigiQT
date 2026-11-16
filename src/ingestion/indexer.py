import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from rank_bm25 import BM25Okapi

from src.config import POLICY_PDF_PATH, CHROMA_PERSIST_DIRECTORY
from src.ingestion.parser import parse_policy_pdf

def tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())

class PolicyKnowledgeBase:
    """
    Hybrid Retrieval Knowledge Base for HDFC ERGO Optima Secure Policy.
    Combines:
      1. ChromaDB (Dense Vector Search with page/section/clause metadata)
      2. BM25 (Sparse Keyword Search for medical terms, waiting period numbers, clause codes)
      3. Reciprocal Rank Fusion (RRF) for optimal ranking
    """
    _instance = None

    def __init__(self, pdf_path: str = POLICY_PDF_PATH, persist_dir: str = CHROMA_PERSIST_DIRECTORY, force_reindex: bool = False):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.collection_name = "hdfc_optima_secure"
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        if force_reindex:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print(f"[PolicyKnowledgeBase] Deleted existing collection '{self.collection_name}' for reindexing.")
            except Exception:
                pass

        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._initialize()

    def _initialize(self):
        # Check if already populated in Chroma
        existing_count = self.collection.count()
        if existing_count == 0:
            print(f"[PolicyKnowledgeBase] Indexing policy document: {self.pdf_path}")
            self.chunks = parse_policy_pdf(self.pdf_path)
            
            # Prepare data for Chroma
            ids = [f"chunk_{i}" for i in range(len(self.chunks))]
            documents = [c["text"] for c in self.chunks]
            metadatas = [
                {
                    "page": int(c["page"]),
                    "section": str(c["section"]),
                    "clause": str(c["clause"])
                }
                for c in self.chunks
            ]
            
            # Batch add to Chroma
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            print(f"[PolicyKnowledgeBase] Successfully indexed {len(self.chunks)} chunks in ChromaDB.")
        else:
            # Reconstruct chunks list from Chroma for BM25
            results = self.collection.get(include=["documents", "metadatas"])
            self.chunks = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                self.chunks.append({
                    "text": doc,
                    "page": meta["page"],
                    "section": meta["section"],
                    "clause": meta["clause"]
                })
            print(f"[PolicyKnowledgeBase] Loaded {len(self.chunks)} chunks from persistent ChromaDB.")

        # Build BM25 index over all chunks
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query: str, top_k: int = 5, max_per_clause: int = 2) -> List[Dict[str, Any]]:
        """
        Executes hybrid search combining Dense (Chroma) and Sparse (BM25) via RRF.
        Applies Diversity Capping (max_per_clause) to prevent 'Cluster Crowding'
        where one repetitive section crowds out base benefits or exclusions.
        """
        candidate_count = min(35, len(self.chunks))

        # 1. Dense retrieval (candidate pool)
        dense_results = self.collection.query(query_texts=[query], n_results=candidate_count)
        dense_ranks = {}
        if dense_results and "documents" in dense_results and dense_results["documents"]:
            for rank, (doc, meta) in enumerate(zip(dense_results["documents"][0], dense_results["metadatas"][0])):
                dense_ranks[doc] = {
                    "rank": rank + 1,
                    "meta": meta,
                    "text": doc
                }

        # 2. Sparse (BM25) retrieval (candidate pool)
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_count]
        sparse_ranks = {}
        for rank, idx in enumerate(sorted_indices):
            chunk = self.chunks[idx]
            sparse_ranks[chunk["text"]] = {
                "rank": rank + 1,
                "meta": {
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "clause": chunk["clause"]
                },
                "text": chunk["text"],
                "score": bm25_scores[idx]
            }

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_const = 60
        all_docs = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))

        for doc in all_docs:
            score = 0.0
            meta = None
            if doc in dense_ranks:
                score += 1.0 / (k_const + dense_ranks[doc]["rank"])
                meta = dense_ranks[doc]["meta"]
            if doc in sparse_ranks:
                score += 1.0 / (k_const + sparse_ranks[doc]["rank"])
                meta = meta or sparse_ranks[doc]["meta"]

            rrf_scores[doc] = {
                "score": score,
                "meta": meta,
                "text": doc
            }

        # Sort by RRF score descending
        ranked_chunks = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        # 4. Diversity Capping (Prevents single-clause cluster crowding)
        formatted_results = []
        clause_counts = {}

        for item in ranked_chunks:
            meta = item["meta"]
            clause_name = meta.get("clause", "Unknown")
            
            if clause_counts.get(clause_name, 0) < max_per_clause:
                clause_counts[clause_name] = clause_counts.get(clause_name, 0) + 1
                formatted_results.append({
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "clause": meta.get("clause"),
                    "citation": f"Page: {meta.get('page')}\nSection: {meta.get('section')}\nClause: {meta.get('clause')}",
                    "text": item["text"],
                    "score": item["score"]
                })
            
            if len(formatted_results) >= top_k:
                break

        return formatted_results

# Global singleton accessor
_kb_instance: Optional[PolicyKnowledgeBase] = None

def get_knowledge_base() -> PolicyKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = PolicyKnowledgeBase()
    return _kb_instance

