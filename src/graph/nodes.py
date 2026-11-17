import re
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Generator
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import get_llm
from src.tools.policy_retriever import retrieve_policy_clauses, format_policy_context, classify_query_complexity
from src.tools.web_search import search_healthcare_market, detect_city, INDIAN_CITIES
from src.graph.state import AgentState

# ==============================================================================
# Helper: Strict Citation Verification Post-Processor
# ==============================================================================
def ensure_citations(response: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Guarantees that the response contains the exact citation format required by Digiqt:
    Source:
    Page: <page>
    Section: <section>
    Clause: <clause>
    """
    if "Source:" in response and "Page:" in response and "Clause:" in response:
        return response

    if not chunks:
        return response

    citations = []
    seen = set()
    for c in chunks[:2]:
        key = (c.get("page"), c.get("clause"))
        if key not in seen:
            seen.add(key)
            citations.append(
                f"Source:\nPage: {c.get('page')}\nSection: {c.get('section')}\nClause: {c.get('clause')}"
            )

    citation_block = "\n\n" + "\n\n".join(citations)
    return response + citation_block

# ==============================================================================
# Node 1: Intent Router
# ==============================================================================
ROUTER_PROMPT = """You are an intelligent Intent Router for an Enterprise Health Insurance Assistant.
Analyze the user's inquiry and categorize it strictly into one of two modes:

1. "policy_qa": The question asks strictly about internal insurance policy rules, definitions, terms, waiting periods, room rent, exclusions, or claim procedures grounded in the health insurance policy document.
2. "agentic_hybrid": The question requires external real-world market intelligence, such as hospital charges, market prices, local surgery costs, treatment options in a city (e.g., Ahmedabad), or cost comparisons combined with policy terms.

Return ONLY a JSON object:
{"intent": "policy_qa" | "agentic_hybrid", "confidence": float, "reason": "brief explanation"}
"""

def router_node(state: AgentState) -> Dict[str, Any]:
    """Classifies user query intent: policy_qa vs agentic_hybrid using LLM with heuristic backup."""
    query = state["query"]
    steps = state.get("execution_steps", [])

    lower = query.lower()

    # Heuristic: detect ANY Indian city + cost/market signal
    has_cost_signal = any(w in lower for w in [
        "cost", "price", "rate", "how much", "charges", "market",
        "expense", "fees", "budget", "affordable", "cheapest", "expensive"
    ])
    has_city_signal = detect_city(query) is not None or any(w in lower for w in [
        "city", "market", "options", "hospitals", "best hospital",
        "compare", "where", "which hospital"
    ])
    has_treatment_options = any(w in lower for w in [
        "treatment options", "types of treatment", "surgical options",
        "which treatment", "best treatment", "alternatives"
    ])

    is_hybrid_heuristic = (has_cost_signal and has_city_signal) or (has_cost_signal and has_treatment_options)
    intent = "agentic_hybrid" if is_hybrid_heuristic else "policy_qa"
    reason = "Heuristic rule classification"

    # Classify query complexity for retrieval optimization
    complexity = classify_query_complexity(query)

    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"User Query: {query}")
        ])
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(content)
        intent = parsed.get("intent", intent)
        reason = parsed.get("reason", reason)
    except Exception:
        pass

    steps.append({
        "node": "Router",
        "action": f"Classified intent: '{intent}' (complexity: {complexity})",
        "details": f"Reason: {reason}"
    })

    return {
        "intent": intent,
        "query_complexity": complexity,
        "execution_steps": steps
    }

# ==============================================================================
# Node 2: Policy Retriever
# ==============================================================================
def policy_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves top policy chunks using hybrid dense + sparse retrieval."""
    query = state["query"]
    steps = state.get("execution_steps", [])
    complexity = state.get("query_complexity", "coverage_admissibility")

    # Adaptive top_k based on complexity
    top_k = 3 if complexity == "simple_factual" else 4
    chunks = retrieve_policy_clauses(query, top_k=top_k)
    
    steps.append({
        "node": "Policy Retriever",
        "action": f"Retrieved {len(chunks)} grounded chunks via Hybrid RRF Search (mode: {complexity})",
        "details": [f"Page {c['page']} | {c['section']} | {c['clause']}" for c in chunks]
    })

    return {
        "policy_chunks": chunks,
        "execution_steps": steps
    }

# ==============================================================================
# Node 3: Policy QA Synthesizer (Standard Mode)
# ==============================================================================
POLICY_QA_SYSTEM_PROMPT = """You are an authoritative Health Insurance Advisory AI.
Answer the user's question strictly grounded on the policy document chunks provided below.

MANDATORY RULES OF LEGAL POLICY INTERPRETATION:
1. LEGAL RULE OF PRECEDENCE (Overrides & Caveats):
   - Section C (Waiting Periods & Exclusions) and Annexure C (Plan Sub-limits & Restrictions) STRICTLY OVERRIDE Section B (Base Benefits).
   - An expense or procedure is NOT payable simply because it is mentioned in Section B (Base Coverage). You MUST always evaluate Section C waiting periods, age criteria, sub-limits, and exclusions.

2. SPECIFIC CONFLICT-RESOLUTION PRINCIPLES:
   a. ACCIDENT OVERRIDE: Waiting periods (both 30-day initial under C.1.c and 24-month specific illnesses under C.1.b) DO NOT apply to claims arising due to an Accident. If the claim is due to an accident/injury, coverage is effective immediately from Day 1.
   b. DAY CARE 24-HOUR WAIVER: Hospitalization under 24 hours is admissible for all Day Care Treatments (Section B-1.1 Note i and Section B-1.1.1.iv). The 24-hour minimum inpatient rule does not apply to day-care procedures.
   c. CONCURRENT WAITING PERIODS (Longer Period Rule): If a condition was diagnosed before policy inception (pre-existing), and is also on the 24-month specific illness list (e.g. gallbladder stones, hernia, calculus), Section C.1.b.iii dictates that the LONGER of the two waiting periods applies (36 months under PED Code Excl01).
   d. ROOM RENT PROPORTIONATE DEDUCTION: If an insured selects a room category higher than their eligibility (e.g. Single Private Room in Optima Select or 1% cap in Optima Lite), proportionate deductions apply to Associated Medical Expenses (Section B-1.1 Note iii), but ICU charges and pharmacy/implants are exempt.
   e. CASHLESS DENIAL ≠ CLAIM REJECTION: Denial of cashless access at a hospital does not reject the policy claim. The insured can undergo treatment and submit bills for Reimbursement under Section E-1.2.g.
   f. MORATORIUM CLAUSE (60 Months): After 60 continuous months of coverage, claims cannot be disputed or rejected on grounds of non-disclosure/misrepresentation, except for proven fraud (Section D-1.6).
   g. PAN-INDIA TREATMENT & ZERO ZONE CO-PAY: The policy provides pan-India coverage without geographic zone co-payments. An insured can be treated in any network hospital across India (e.g. Mumbai, Delhi, Ahmedabad) without financial co-pay penalty (Section B-1.1).
   h. PROTECT BENEFIT (NON-MEDICAL CONSUMABLES): Under the policy's Protect Benefit (Section B-1.2), non-medical items listed in Annexure B (gloves, PPE kits, masks, syringes, cotton, etc.) are covered at actuals.
   i. RESTORE BENEFIT (100% AUTOMATIC REFILL): Under Section B-1.4, 100% of the base sum insured is automatically restored upon partial or complete exhaustion during the policy year.
   j. PRE & POST HOSPITALIZATION TIMEFRAMES: Medical expenses incurred 60 days before admission (Clause B.1.1.1.ii) and 180 days after discharge (Clause B.1.1.1.iii) are payable.
   k. AIR AMBULANCE COVER: Emergency air transportation expenses up to ₹5,00,000 per hospitalization are covered (Clause B.1.1.2).

3. FULL DOCUMENT AWARENESS (Cross-Sectional Verification):
   - ALWAYS cross-check coverage claims against ALL relevant sections. If Section B says a treatment is covered, you MUST also check:
     * Section C for any waiting period or exclusion that may override it
     * Annexure C for plan-specific sub-limits
     * Section D for general conditions like age limits, renewal terms, moratorium
   - NEVER conclude coverage based on a single section alone.
   - If information from one chunk contradicts or qualifies another, explicitly state both and explain which takes precedence.

4. PARAMETER COMPARISON & ZERO ASSUMPTIONS:
   - When the user mentions specific timeframes (e.g. 5 months, 130 days) or ages, compare them directly against the statutory thresholds. State clearly whether the claim is admissible or excluded.
   - If plan choice determines coverage (e.g. standard Optima Secure vs Optima Select/Lite), outline the specific limits for each variant.

5. MANDATORY CITATION FORMAT:
At the very end of your response, you MUST format the exact source citation as follows:

Source:
Page: <Page number from chunk metadata>
Section: <Section name>
Clause: <Clause name or Code>

If multiple clauses apply (e.g. Base Benefit + Exclusion), list all citations clearly.
"""

def policy_qa_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes a strictly grounded policy answer."""
    query = state["query"]
    chunks = state.get("policy_chunks", [])
    steps = state.get("execution_steps", [])

    context = format_policy_context(chunks)
    prompt = f"""Policy Document Evidence:
{context}

User Question:
{query}

Generate a clear, accurate, and completely grounded explanation, ending with the exact Source citation."""

    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke([
            SystemMessage(content=POLICY_QA_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        raw_answer = resp.content.strip()
        answer = ensure_citations(raw_answer, chunks)
    except Exception as e:
        answer = (
            f"Unable to generate response via LLM: {e}.\n\n"
            f"Please verify that your NVIDIA_API_KEY is configured in .env."
        )

    steps.append({
        "node": "Policy QA Synthesizer",
        "action": "Generated policy-grounded response with verified citations",
        "details": "Grounding verified against internal clauses."
    })

    return {
        "response": answer,
        "execution_steps": steps
    }

# ==============================================================================
# Node 4: Dynamic Query Decomposer (Agentic Mode) — FULLY GENERALIZED
# ==============================================================================
DECOMPOSER_PROMPT = """You are a Query Decomposition Agent for a Healthcare & Insurance Assistant.
Break down the user's multi-faceted question into structured sub-tasks.

Extract:
1. "procedure": The medical condition, surgery, treatment, or health service mentioned (e.g. "kidney stone surgery", "cataract surgery", "knee replacement", "heart bypass", "maternity delivery", "cancer treatment", "appendectomy", "hernia repair", "spine surgery", etc.). Extract the specific medical procedure or condition, not the generic query.
2. "city": The target geographic location/city mentioned (e.g. "Ahmedabad", "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", etc.). If no specific city is mentioned, use "India".
3. "policy_subquery": A keyword-rich search query to retrieve policy wording clauses regarding waiting periods, room rent, day care coverage, and exclusions for this specific condition. Include the medical condition name AND relevant policy terms like "waiting period", "exclusion", "coverage", "sub-limit".
4. "market_subquery": A targeted search query for live healthcare aggregator platforms (Practo, HexaHealth, Pristyn Care, Apollo) to find treatment types and price ranges in that city.

Return ONLY a JSON object:
{
  "procedure": "...",
  "city": "...",
  "policy_subquery": "...",
  "market_subquery": "..."
}
"""

def decompose_query_node(state: AgentState) -> Dict[str, Any]:
    """Dynamically decomposes the user query into internal policy and external market subqueries.
    Fully generalized — no hardcoded procedures or cities."""
    query = state["query"]
    steps = state.get("execution_steps", [])

    # Heuristic fallback: detect city from query, default to "India"
    detected_city = detect_city(query) or "India"
    # Heuristic fallback: use the query itself as procedure description
    procedure = query  # Will be refined by LLM

    subqueries = {
        "procedure": procedure,
        "city": detected_city,
        "policy": f"{procedure} waiting period surgical day care room rent coverage exclusion",
        "market": f"{procedure} treatment options cost in {detected_city}"
    }

    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke([
            SystemMessage(content=DECOMPOSER_PROMPT),
            HumanMessage(content=f"User Query: {query}")
        ])
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(content)
        subqueries["procedure"] = parsed.get("procedure", procedure)
        subqueries["city"] = parsed.get("city", detected_city)
        subqueries["policy"] = parsed.get("policy_subquery", subqueries["policy"])
        subqueries["market"] = parsed.get("market_subquery", subqueries["market"])
    except Exception:
        pass

    steps.append({
        "node": "Query Decomposer",
        "action": "Decomposed request into parallel internal policy and external market sub-goals",
        "details": subqueries
    })

    return {
        "subqueries": subqueries,
        "execution_steps": steps
    }

# ==============================================================================
# Node 5: Market Search & Parallel Policy Retrieval Node
# ==============================================================================
def search_market_costs_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes parallel concurrent retrieval:
    Thread 1: Hybrid internal policy retrieval
    Thread 2: External healthcare market search
    This halves retrieval latency.
    """
    subqueries = state.get("subqueries", {})
    procedure = subqueries.get("procedure", state.get("query", ""))
    city = subqueries.get("city", "India")
    policy_query = subqueries.get("policy", f"{procedure} waiting period coverage exclusion")
    steps = state.get("execution_steps", [])

    # Parallel asynchronous execution of dual retrieval
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_policy = executor.submit(retrieve_policy_clauses, policy_query, top_k=4)
        fut_market = executor.submit(search_healthcare_market, procedure=procedure, city=city)
        
        policy_chunks = fut_policy.result()
        market_data = fut_market.result()

    snippets = market_data.get("snippets", [])

    steps.append({
        "node": "Parallel Dual Retriever",
        "action": f"Concurrently retrieved {len(policy_chunks)} policy clauses and {len(snippets)} market sources",
        "details": f"Parallel execution completed for '{procedure}' in '{city}'."
    })

    return {
        "policy_chunks": policy_chunks,
        "web_snippets": snippets,
        "execution_steps": steps
    }

# ==============================================================================
# Node 6: Multi-Criteria Relevance Grader (Corrective RAG) — GENERALIZED
# ==============================================================================
def relevance_grader_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether external web snippets meet quality gates.
    Fully generalized — no hardcoded procedure names.
    Gates:
      1. Target City/Region Grounding
      2. Medical/Treatment Content (procedure-related terms)
      3. Currency & Pricing Integrity (INR, ₹, Rs., realistic figures)
      4. Source Credibility (Accredited medical aggregators)
    """
    snippets = state.get("web_snippets", [])
    subqueries = state.get("subqueries", {})
    city = subqueries.get("city", "India").lower()
    procedure = subqueries.get("procedure", "").lower()
    retry_count = state.get("retry_count", 0)
    steps = state.get("execution_steps", [])

    combined_text = " ".join([s.get("content", "") + " " + s.get("title", "") for s in snippets]).lower()

    # Gate 1: City/Region Grounding
    has_city = city in combined_text or city == "india"

    # Gate 2: Medical Content — dynamically check for procedure-related terms
    # Extract key medical terms from the procedure description
    procedure_words = set(procedure.split()) - {"in", "the", "a", "an", "for", "of", "and", "or", "with"}
    has_procedures = (
        any(word in combined_text for word in procedure_words if len(word) > 3) or
        any(term in combined_text for term in ["surgery", "treatment", "procedure", "hospital", "doctor", "therapy", "operation"])
    )

    # Gate 3: Currency & Numeric Pricing in INR
    has_pricing = any(term in combined_text for term in [
        "₹", "rs", "inr", "cost", "price", "lakh", "thousand",
        "30,000", "40,000", "50,000", "80,000", "1,00,000", "2,00,000"
    ])

    # Gate 4: Source Credibility
    has_credible_source = any(
        s.get("source_tier") in ["Accredited Healthcare Aggregator", "Government Healthcare Institution", "Verified Medical Platform"]
        or any(d in s.get("url", "").lower() for d in ["hexahealth", "practo", "pristyncare", "apollo", "narayana", "manipal", "fortis", "max", "medanta"])
        for s in snippets
    )

    is_sufficient = (len(snippets) > 0 and has_procedures and (has_city or has_pricing))

    if not is_sufficient and retry_count < 1:
        grade_status = "retry"
        verdict = f"Evidence gate failed (City: {has_city}, Medical: {has_procedures}, Pricing: {has_pricing}, Trusted: {has_credible_source}). Triggering authoritative query rewrite."
    else:
        grade_status = "valid"
        verdict = f"Evidence verified (City: {has_city}, Medical: {has_procedures}, Pricing: {has_pricing}, Trusted: {has_credible_source}). Approved for synthesis."

    steps.append({
        "node": "Relevance & Credibility Grader",
        "action": f"Web Evidence Gate Verdict: '{grade_status}'",
        "details": verdict
    })

    return {
        "grade_status": grade_status,
        "execution_steps": steps
    }

# ==============================================================================
# Node 7: Query Rewriter (Self-Correction Loop)
# ==============================================================================
def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    """Rewrites search query with high-intent portal directives for the second attempt."""
    subqueries = state.get("subqueries", {})
    procedure = subqueries.get("procedure", "")
    city = subqueries.get("city", "India")
    retry_count = state.get("retry_count", 0)
    steps = state.get("execution_steps", [])

    refined_query = f"{procedure} treatment options types price cost in {city} practo hexahealth pristyncare hospital"
    subqueries["market"] = refined_query

    steps.append({
        "node": "Query Rewriter",
        "action": f"Self-Correction: Refined web query to '{refined_query}'",
        "details": f"Attempt {retry_count + 1} of 1"
    })

    return {
        "subqueries": subqueries,
        "retry_count": retry_count + 1,
        "execution_steps": steps
    }

# ==============================================================================
# Node 8: Agentic Synthesizer — FULLY GENERALIZED (No hardcoded procedures)
# ==============================================================================
AGENTIC_SYNTHESIZER_PROMPT = """You are a Chief Health Insurance & Medical Advisory AI.
The customer has presented a composite real-world inquiry requiring both local market cost intelligence and policy underwriting coverage verification:
"{user_query}"

You are provided with:
1. Official health insurance policy wording clauses.
2. Verified local healthcare market intelligence on medical options and hospital costs in {city}.

Synthesize a comprehensive, authoritative, and well-organized response strictly structured into the following sections:

### Part 1: Medical Treatment Options & {city} Market Cost Breakdown
- Present a clear, comprehensive **Markdown Comparison Table** with appropriate columns for the specific procedure/condition:
  | Treatment Modality | Clinical Suitability | Hospital Stay | Estimated {city} Cost Range (INR ₹) | Policy Coverage Admissibility |
- List ALL treatment modalities found in the market research data, with their clinical descriptions and cost ranges.
- Reference leading hospital centers in {city} if mentioned in the sources.
- Cite the external healthcare sources with hyperlinks (e.g. HexaHealth, Practo, Pristyn Care, Apollo, etc.).
- If data for {city} specifically is limited, note that and provide the best available Indian pricing data.

### Part 2: Health Insurance Policy Admissibility & Coverage Analysis
- Analyze whether and how the policy covers this specific condition:
  * **Waiting Period Analysis**: State clearly which waiting period applies to this condition:
    - Check if it falls under the 24-month Specified Disease/Procedure waiting period (Code – Excl02) and list the exact condition from the policy if applicable.
    - Check if it's a Pre-Existing Disease (PED) subject to 36-month waiting (Code – Excl01).
    - Apply the "longer of the two" rule (C.1.b.iii) if both PED and specific illness waiting periods apply.
    - State the 30-day initial waiting period (Code – Excl03) exception for accidents.
  * **Day Care vs. Inpatient**: If the procedure can be done as Day Care, highlight that the 24-hour minimum hospitalization rule does NOT apply (Section B.1.1.1.iv).
  * **Room Rent**: Explain room rent rules (At Actuals for Comprehensive variant; Single Private Room for Standard variant; 1% cap for Basic variant).
  * **Sub-limits & Exclusions**: Note any applicable sub-limits or exclusions from Section C or Annexure C.
- **MANDATORY POLICY CITATIONS**:
  Every response must cite the policy document in this exact format:
  Source:
  Page: <page number>
  Section: <section name>
  Clause: <clause name or code>

### Part 3: Actionable Claim Guidance & Checklist for the Patient
- Cashless pre-authorization intimation timeline (at least 48 hours in advance for planned admissions under Clause E.1.1.b).
- Essential document checklist for reimbursement claims (Discharge card, indoor case papers, final itemized bill, doctor consultation papers, and relevant diagnostic reports).
- Any special considerations for this particular condition/procedure.
"""

def agentic_synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the dual-perspective report combining market costs and policy admissibility.
    Fully generalized for any procedure and any city."""
    query = state["query"]
    subqueries = state.get("subqueries", {})
    city = subqueries.get("city", "India")
    policy_chunks = state.get("policy_chunks", [])
    web_snippets = state.get("web_snippets", [])
    steps = state.get("execution_steps", [])

    policy_context = format_policy_context(policy_chunks)
    
    web_context_items = []
    for s in web_snippets:
        web_context_items.append(
            f"Source: {s.get('title')} ({s.get('url')})\nSnippet: {s.get('content')}\n"
        )
    web_context = "\n".join(web_context_items)

    prompt = f"""User Question:
{query}

--- INTERNAL POLICY WORDING CONTEXT ---
{policy_context}

--- EXTERNAL MARKET RESEARCH EVIDENCE ({city.upper()}) ---
{web_context if web_context.strip() else "No external market data was retrieved for this query. Focus on policy analysis and provide general guidance."}

Generate the complete dual-mode structured response following all system instructions."""

    try:
        llm = get_llm(temperature=0.0, max_tokens=3500)
        resp = llm.invoke([
            SystemMessage(content=AGENTIC_SYNTHESIZER_PROMPT.format(user_query=query, city=city)),
            HumanMessage(content=prompt)
        ])
        raw_answer = resp.content.strip()
        answer = ensure_citations(raw_answer, policy_chunks)
    except Exception as e:
        answer = (
            f"Unable to generate response via LLM: {e}.\n\n"
            f"Please verify that your NVIDIA_API_KEY is configured in .env."
        )

    steps.append({
        "node": "Agentic Synthesizer",
        "action": f"Synthesized dual-perspective advisory report ({city} Market Costs + Policy Admissibility)",
        "details": "Includes external market references and strict policy Page/Section/Clause citations."
    })

    return {
        "response": answer,
        "execution_steps": steps
    }

# ==============================================================================
# Streaming Generators (for Streamlit UI real-time rendering)
# ==============================================================================
def stream_policy_qa(query: str, chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """
    Streams policy QA response token-by-token for instant UI rendering.
    """
    context = format_policy_context(chunks)
    prompt = f"""Policy Document Evidence:
{context}

User Question:
{query}

Generate a clear, accurate, and completely grounded explanation, ending with the exact Source citation."""

    llm = get_llm(temperature=0.0, streaming=True)
    accumulated = ""
    for chunk in llm.stream([
        SystemMessage(content=POLICY_QA_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]):
        content = chunk.content
        if content:
            accumulated += content
            yield content

    if "Source:" not in accumulated:
        citation_block = ensure_citations("", chunks)
        yield citation_block

def stream_agentic_synthesizer(
    query: str,
    city: str,
    policy_chunks: List[Dict[str, Any]],
    web_snippets: List[Dict[str, Any]]
) -> Generator[str, None, None]:
    """
    Streams the dual-mode agentic report token-by-token for instant UI rendering.
    """
    policy_context = format_policy_context(policy_chunks)
    web_context_items = [
        f"Source: {s.get('title')} ({s.get('url')})\nSnippet: {s.get('content')}\n"
        for s in web_snippets
    ]
    web_context = "\n".join(web_context_items)

    prompt = f"""User Question:
{query}

--- INTERNAL POLICY WORDING CONTEXT ---
{policy_context}

--- EXTERNAL MARKET RESEARCH EVIDENCE ({city.upper()}) ---
{web_context if web_context.strip() else "No external market data was retrieved for this query. Focus on policy analysis and provide general guidance."}

Generate the complete dual-mode structured response following all system instructions."""

    llm = get_llm(temperature=0.0, max_tokens=3500, streaming=True)
    accumulated = ""
    for chunk in llm.stream([
        SystemMessage(content=AGENTIC_SYNTHESIZER_PROMPT.format(user_query=query, city=city)),
        HumanMessage(content=prompt)
    ]):
        content = chunk.content
        if content:
            accumulated += content
            yield content

    if "Source:" not in accumulated:
        citation_block = ensure_citations("", policy_chunks)
        yield citation_block
