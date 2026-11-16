from typing import List, Dict, Any, Optional, Set
from src.ingestion.indexer import get_knowledge_base

# ==============================================================================
# Medical Synonym Expansion Map
# ==============================================================================
# Maps common terms to their clinical/policy-document equivalents so that
# a user asking about "heart attack" also retrieves "myocardial infarction",
# or "hip replacement" also catches "arthroplasty".
MEDICAL_SYNONYM_MAP: Dict[str, List[str]] = {
    "heart attack": ["myocardial infarction", "cardiac", "coronary artery", "angioplasty", "bypass"],
    "cardiac": ["heart", "coronary", "angioplasty", "bypass", "myocardial"],
    "kidney stone": ["calculus", "urogenital", "lithotripsy", "renal calculi", "nephrolithiasis"],
    "gallbladder": ["cholecystitis", "cholecystectomy", "biliary", "gall bladder"],
    "cataract": ["phacoemulsification", "iol", "intraocular lens", "ophthalmology"],
    "cancer": ["malignant", "oncology", "tumour", "tumor", "chemotherapy", "radiation"],
    "maternity": ["pregnancy", "delivery", "caesarean", "c-section", "childbirth", "obstetric", "newborn"],
    "knee replacement": ["arthroplasty", "knee joint", "total knee", "orthopedic"],
    "hip replacement": ["arthroplasty", "hip joint", "total hip", "orthopedic"],
    "hernia": ["inguinal", "umbilical", "hiatal", "laparoscopic repair", "hernioplasty"],
    "diabetes": ["diabetic", "insulin", "blood sugar", "hyperglycemia", "hba1c"],
    "hypertension": ["blood pressure", "bp", "high bp", "antihypertensive"],
    "thyroid": ["hypothyroidism", "hyperthyroidism", "thyroidectomy", "goiter"],
    "spine": ["spinal", "vertebral", "disc", "laminectomy", "spondylosis", "back surgery"],
    "dental": ["tooth", "teeth", "oral surgery", "root canal", "extraction"],
    "mental health": ["psychiatric", "depression", "anxiety", "psychological", "counseling"],
    "dialysis": ["hemodialysis", "peritoneal", "renal failure", "kidney failure", "chronic kidney"],
    "transplant": ["organ transplant", "donor", "recipient", "graft"],
    "stroke": ["cerebrovascular", "brain hemorrhage", "paralysis", "thrombosis"],
    "appendix": ["appendicitis", "appendectomy", "appendicular"],
}

# ==============================================================================
# Simple-Factual Query Detector
# ==============================================================================
# Definition requests and simple lookups that do not require cross-sectional
# conflict-resolution expansions (saves ~50% retrieval latency and avoids noise).
DEFINITION_PATTERNS = [
    "definition of", "define", "meaning of", "what is meant by",
    "what is the definition", "what does the policy define",
]

SIMPLE_FACTUAL_PATTERNS = [
    "what is", "what are", "what does", "explain", "list of", "list the",
    "how many", "who is", "when was", "where is",
]

# Actionable coverage questions that require full cross-sectional analysis
ACTIONABLE_COVERAGE_QUESTIONS = [
    "is it covered", "is this covered", "am i covered", "will it be covered",
    "will insurance cover", "can i claim", "will my claim", "is covered",
    "waiting period for", "waiting period on", "how long is the waiting",
    "room rent limit", "can i choose", "what happens if", "is there a cap",
    "proportionate deduction", "cashless denied", "claim rejected",
    "days ago", "months ago", "years ago",
]

def classify_query_complexity(query: str) -> str:
    """
    Classifies query into complexity tiers to determine retrieval strategy:
    - 'simple_factual': Definition lookups, document references → direct retrieval only
    - 'coverage_admissibility': Coverage questions → cross-sectional expansion needed
    """
    lower = query.lower()

    # Explicit definition requests are always simple factual
    if any(dp in lower for dp in DEFINITION_PATTERNS):
        return "simple_factual"

    # Actionable coverage queries that require cross-sectional analysis
    if any(q in lower for q in ACTIONABLE_COVERAGE_QUESTIONS):
        return "coverage_admissibility"

    # General coverage action signals vs simple lookups
    has_coverage_action = any(sig in lower for sig in [
        "covered", "payable", "admissible", "reject", "denied",
        "accident exception", "moratorium", "sublimit", "reimbursement"
    ])
    is_simple = any(pat in lower for pat in SIMPLE_FACTUAL_PATTERNS)

    if is_simple and not has_coverage_action:
        return "simple_factual"
    return "coverage_admissibility"


# ==============================================================================
# Domain-specific trigger keywords mapped to critical override sections
# ==============================================================================
CROSS_SECTIONAL_TRIGGERS = [
    # 1. Accident Exception (Overrides 30-day and 24-month waiting periods)
    (["accident", "injury", "fracture", "fall", "emergency trauma"], 
     "accident exception waiting period waived not applicable Section C.1.b.i C.1.c.i"),

    # 2. Day Care Exception (Overrides 24-hour hospitalization minimum)
    (["day care", "daycare", "same day", "less than 24", "6 hours", "12 hours", "dialysis", "chemotherapy", "cataract"],
     "Day Care Treatment 24 consecutive hours time limit shall not apply Note i Section B.1.1.1.iv"),

    # 3. Pre-existing vs Specific Illness Clash (Longer period applies under C.1.b.iii)
    (["already had", "pre-existing", "prior to", "before buying", "diagnosed before", "130 days", "ped"],
     "Pre-Existing Diseases Code Excl01 36 months longer of the two waiting periods shall apply Section C.1.b.iii"),

    # 4. Room Rent Upgrade & Proportionate Deductions
    (["room rent", "single private", "proportionate", "associated medical expenses", "shared room", "15000"],
     "Proportionate deduction on Room Rent Associated Medical Expenses Note iii Section B.1.1 Clause 2.13 Annexure C"),

    # 5. Cashless Denial vs Reimbursement
    (["cashless denied", "denial of cashless", "pre-authorization rejected", "cashless rejected"],
     "denial of cashless access submit claim documents for reimbursement Section E.1.2.g"),

    # 6. Moratorium Period
    (["moratorium", "60 months", "5 years", "contestable", "non-disclosure"],
     "Moratorium Period sixty continuous months contestable Section D.1.6"),

    # 7. Age-related exclusions & limitations
    (["age", "years old", "elderly", "senior citizen", "age limit", "age 65", "age 60"],
     "age limit entry age eligibility renewal lifetime Section D General Terms"),

    # 8. Maternity / Newborn coverage conditions
    (["maternity", "pregnancy", "delivery", "newborn", "baby", "caesarean", "c-section"],
     "maternity expenses newborn baby delivery complications Section B Optional Cover waiting period"),

    # 9. AYUSH treatment coverage
    (["ayush", "ayurveda", "homeopathy", "unani", "siddha", "naturopathy"],
     "AYUSH treatment Ayurveda Yoga Unani Siddha Homeopathy Section B coverage"),

    # 10. Organ Transplant / Donor coverage
    (["transplant", "organ", "donor", "recipient"],
     "organ transplant donor expenses recipient coverage Section B"),
]


def expand_query_with_synonyms(query: str) -> List[str]:
    """
    Generates additional search queries by expanding medical terms with
    their clinical synonyms. Returns [original_query, expanded_query1, ...].
    """
    lower = query.lower()
    expansions = set()
    
    for term, synonyms in MEDICAL_SYNONYM_MAP.items():
        if term in lower:
            # Add 2-3 most relevant synonyms to create expanded queries
            for syn in synonyms[:3]:
                if syn not in lower:
                    expansions.add(syn)
    
    if not expansions:
        return [query]
    
    # Build one expanded query combining top synonyms
    expansion_str = " ".join(list(expansions)[:4])
    expanded_query = f"{query} {expansion_str}"
    return [query, expanded_query]


def retrieve_policy_clauses(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieves policy clauses matching the query using hybrid (Chroma + BM25) retrieval.

    Retrieval strategy adapts based on query complexity:
    - simple_factual: Direct hybrid search only (fast path)
    - coverage_admissibility: Full cross-sectional expansion with override triggers

    Also applies medical synonym expansion for better recall on clinical terms.
    """
    kb = get_knowledge_base()
    lower_q = query.lower()
    complexity = classify_query_complexity(query)

    # 1. Primary search (always runs)
    queries = expand_query_with_synonyms(query)
    primary_results = kb.hybrid_search(query=queries[0], top_k=top_k)

    # If expanded queries exist, merge results from them too
    if len(queries) > 1:
        for eq in queries[1:]:
            extra = kb.hybrid_search(query=eq, top_k=3)
            primary_results.extend(extra)

    # FAST PATH: Simple factual queries skip cross-sectional expansion
    if complexity == "simple_factual":
        return _deduplicate(primary_results)[:top_k]

    # 2. FULL PATH: Check for domain-specific conflict/exception triggers
    secondary_results = []
    for keywords, override_query in CROSS_SECTIONAL_TRIGGERS:
        if any(kw in lower_q for kw in keywords):
            override_hits = kb.hybrid_search(query=override_query, top_k=3)
            secondary_results.extend(override_hits)

    # 3. If standard coverage query, also expand to general waiting periods & exclusions
    is_general_coverage = any(w in lower_q for w in ["covered", "cover", "surgery", "hospitalized", "claim"])
    if is_general_coverage and not secondary_results:
        gen_hits = kb.hybrid_search(query=f"{query} waiting period exclusion sublimit schedule of benefits", top_k=3)
        secondary_results.extend(gen_hits)

    # Merge and deduplicate
    all_results = primary_results + secondary_results
    return _deduplicate(all_results)[:top_k]


def _deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates results by (page, clause, text_snippet)."""
    seen_keys: Set = set()
    merged: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("page"), r.get("clause"), r.get("text", "")[:80])
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(r)
    return merged


def format_policy_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into clean markdown context for LLM prompt injection.
    """
    formatted_sections = []
    for idx, c in enumerate(chunks, 1):
        formatted_sections.append(
            f"--- CHUNK {idx} ---\n"
            f"Source:\n"
            f"Page: {c['page']}\n"
            f"Section: {c['section']}\n"
            f"Clause: {c['clause']}\n\n"
            f"Content:\n{c['text']}\n"
        )
    return "\n\n".join(formatted_sections)
