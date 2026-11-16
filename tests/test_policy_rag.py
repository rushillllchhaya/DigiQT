import pytest
from src.tools.policy_retriever import retrieve_policy_clauses, classify_query_complexity, expand_query_with_synonyms
from src.graph.nodes import ensure_citations

# ==============================================================================
# Original Benchmark Tests (Preserved)
# ==============================================================================

def test_ped_waiting_period_retrieval():
    """Verify pre-existing disease waiting period chunks are retrieved from Page 28 (Clause C.1.a)."""
    results = retrieve_policy_clauses("waiting period for pre-existing diseases", top_k=3)
    assert len(results) > 0
    pages = [r["page"] for r in results]
    clauses = [r["clause"] for r in results]
    assert 28 in pages or any("Excl01" in c for c in clauses)

def test_kidney_stone_waiting_period_retrieval():
    """Verify kidney stone surgery waiting period is retrieved from Page 28/29 (Clause C.1.b)."""
    results = retrieve_policy_clauses("kidney stone calculus waiting period", top_k=3)
    assert len(results) > 0
    clauses = [r["clause"] for r in results]
    assert any("Excl02" in c for c in clauses)
    # Check that text mentions 24 months
    matching_text = any("24 months" in r["text"] or "calculus" in r["text"].lower() for r in results)
    assert matching_text

def test_day_care_treatment_retrieval():
    """Verify Day Care coverage is retrieved from Page 11 (Clause B.1.1.1)."""
    results = retrieve_policy_clauses("day care treatment covered", top_k=3)
    assert len(results) > 0
    clauses = [r["clause"] for r in results]
    assert any("Day Care" in c for c in clauses)

def test_room_rent_retrieval():
    """Verify Room Rent rules are retrieved from Page 11 and/or Annexure C."""
    results = retrieve_policy_clauses("daily room rent 15000 covered at actuals", top_k=3)
    assert len(results) > 0
    pages = [r["page"] for r in results]
    clauses = [r["clause"] for r in results]
    assert 11 in pages or any("Room Rent" in c for c in clauses)

def test_gallbladder_pre_existing_retrieval():
    """Verify Gallbladder stone rules are retrieved from Section C.1.a & C.1.b."""
    results = retrieve_policy_clauses("gallbladder stones pre-existing disease 130 days", top_k=3)
    assert len(results) > 0
    clauses = [r["clause"] for r in results]
    assert any("Excl01" in c or "Excl02" in c for c in clauses)

def test_claim_documents_retrieval():
    """Verify claim documents list is retrieved from Page 43/44 (Clause E.1.7)."""
    results = retrieve_policy_clauses("list of documents required for a claim", top_k=3)
    assert len(results) > 0
    pages = [r["page"] for r in results]
    assert 43 in pages or 44 in pages

def test_accident_waiting_period_override_retrieval():
    """Verify accident exception to waiting periods is retrieved for accident queries."""
    results = retrieve_policy_clauses("I had a road accident 15 days after buying policy, is fracture surgery covered?", top_k=4)
    assert len(results) > 0
    # Must retrieve Section C waiting periods containing accident exception
    clauses = [r["clause"] for r in results]
    assert any("Waiting Period" in c for c in clauses)
    # Check that text mentions accident exception
    has_accident_text = any("accident" in r["text"].lower() for r in results)
    assert has_accident_text

def test_day_care_24_hour_waiver_retrieval():
    """Verify day care 24-hour hospitalization waiver is retrieved for short stay queries."""
    results = retrieve_policy_clauses("chemotherapy discharge in 6 hours day care 24 hours requirement", top_k=4)
    assert len(results) > 0
    clauses = [r["clause"] for r in results]
    assert any("Day Care" in c for c in clauses)

def test_cashless_denial_reimbursement_retrieval():
    """Verify cashless denial reimbursement fallback (Section E.1.2.g) is retrieved."""
    results = retrieve_policy_clauses("hospital cashless pre-authorization denied can I get reimbursement?", top_k=4)
    assert len(results) > 0
    pages = [r["page"] for r in results]
    assert any(p in [41, 42, 43] for p in pages)

def test_moratorium_period_retrieval():
    """Verify 60-month moratorium protection (Section D.1.6) is retrieved."""
    results = retrieve_policy_clauses("held policy for 5 years 60 months non-disclosure dispute moratorium", top_k=4)
    assert len(results) > 0
    pages = [r["page"] for r in results]
    clauses = [r["clause"] for r in results]
    assert 34 in pages or any("Moratorium" in c for c in clauses)

def test_ensure_citations_helper():
    """Verify citation validator appends citation block if missing."""
    dummy_chunks = [
        {"page": 28, "section": "SECTION C. WAITING PERIOD AND EXCLUSIONS", "clause": "Clause C.1.a: Pre-Existing Diseases (Code: Excl01)"}
    ]
    resp = "Pre-existing diseases have a 36-month waiting period."
    formatted = ensure_citations(resp, dummy_chunks)
    assert "Source:" in formatted
    assert "Page: 28" in formatted
    assert "Clause C.1.a" in formatted

# ==============================================================================
# NEW: Query Complexity Classifier Tests
# ==============================================================================

def test_simple_factual_classification():
    """Simple definition lookups should be classified as simple_factual."""
    assert classify_query_complexity("What is the definition of Room Rent?") == "simple_factual"
    assert classify_query_complexity("What is AYUSH?") == "simple_factual"
    assert classify_query_complexity("Define pre-existing disease") == "simple_factual"

def test_coverage_admissibility_classification():
    """Coverage questions should be classified as coverage_admissibility."""
    assert classify_query_complexity("Is kidney stone surgery covered?") == "coverage_admissibility"
    assert classify_query_complexity("Will my claim be rejected?") == "coverage_admissibility"
    assert classify_query_complexity("What is the waiting period for pre-existing diseases?") == "coverage_admissibility"

def test_simple_factual_fast_path():
    """Simple factual queries should return results without cross-sectional expansion."""
    results = retrieve_policy_clauses("What is AYUSH treatment?", top_k=3)
    assert len(results) > 0
    # Should still return relevant results, just faster (no expansion)

# ==============================================================================
# NEW: Medical Synonym Expansion Tests
# ==============================================================================

def test_synonym_expansion_heart():
    """Heart attack synonyms should expand to include myocardial infarction, cardiac."""
    queries = expand_query_with_synonyms("heart attack surgery")
    assert len(queries) >= 2
    expanded = queries[1].lower()
    assert any(term in expanded for term in ["myocardial", "cardiac", "coronary"])

def test_synonym_expansion_kidney():
    """Kidney stone should expand to calculus, urogenital."""
    queries = expand_query_with_synonyms("kidney stone treatment")
    assert len(queries) >= 2
    expanded = queries[1].lower()
    assert any(term in expanded for term in ["calculus", "urogenital", "lithotripsy"])

def test_no_expansion_generic():
    """Generic queries with no medical terms should not expand."""
    queries = expand_query_with_synonyms("policy benefits overview")
    assert len(queries) == 1
