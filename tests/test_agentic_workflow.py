import pytest
from src.graph.state import AgentState
from src.graph.nodes import (
    router_node,
    decompose_query_node,
    relevance_grader_node,
    query_rewriter_node
)
from src.tools.web_search import search_healthcare_market
from src.graph.workflow import get_graph

def test_router_classification_hybrid():
    """Verify that the Ahmedabad surgery cost query is routed to agentic_hybrid."""
    state: AgentState = {
        "query": "If I need kidney stone surgery, how much will it cost, what treatment options are available in the market, and how much does each option typically cost? Tell me for Ahmedabad?",
        "intent": "",
        "subqueries": {},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = router_node(state)
    assert result["intent"] == "agentic_hybrid"

def test_router_classification_policy():
    """Verify standard policy questions route to policy_qa."""
    state: AgentState = {
        "query": "What is the waiting period for pre-existing diseases?",
        "intent": "",
        "subqueries": {},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = router_node(state)
    assert result["intent"] == "policy_qa"

def test_query_decomposition():
    """Verify query decomposer extracts procedure and city."""
    state: AgentState = {
        "query": "If I need kidney stone surgery, how much will it cost... for Ahmedabad?",
        "intent": "agentic_hybrid",
        "subqueries": {},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = decompose_query_node(state)
    subqueries = result["subqueries"]
    assert subqueries["city"] == "Ahmedabad"
    assert "kidney stone" in subqueries["procedure"].lower()

def test_parallel_dual_retriever():
    """Verify parallel dual retriever retrieves both policy chunks and market snippets."""
    from src.graph.nodes import search_market_costs_node
    state: AgentState = {
        "query": "kidney stone surgery Ahmedabad cost",
        "intent": "agentic_hybrid",
        "subqueries": {
            "procedure": "kidney stone surgery",
            "city": "Ahmedabad",
            "policy": "kidney stone calculus waiting period",
            "market": "kidney stone surgery cost in Ahmedabad"
        },
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = search_market_costs_node(state)
    assert len(result["policy_chunks"]) > 0
    assert len(result["web_snippets"]) > 0

def test_relevance_grader_pass():
    """Verify relevance grader passes when snippets have procedure, pricing and city."""
    state: AgentState = {
        "query": "kidney stone surgery Ahmedabad cost",
        "intent": "agentic_hybrid",
        "subqueries": {"procedure": "kidney stone", "city": "Ahmedabad"},
        "policy_chunks": [],
        "web_snippets": [
            {
                "title": "Kidney Stone Cost Ahmedabad",
                "content": "Laser RIRS surgery for kidney stone costs ₹85,000 in Ahmedabad hospitals.",
                "url": "https://example.com"
            }
        ],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    res = relevance_grader_node(state)
    assert res["grade_status"] == "valid"

def test_relevance_grader_retry():
    """Verify relevance grader triggers retry when snippets are missing."""
    state: AgentState = {
        "query": "kidney stone surgery Ahmedabad cost",
        "intent": "agentic_hybrid",
        "subqueries": {"procedure": "kidney stone", "city": "Ahmedabad"},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    res = relevance_grader_node(state)
    assert res["grade_status"] == "retry"

def test_query_rewriter():
    """Verify query rewriter refines search query on retry."""
    state: AgentState = {
        "query": "kidney stone surgery Ahmedabad cost",
        "intent": "agentic_hybrid",
        "subqueries": {"procedure": "kidney stone", "city": "Ahmedabad", "market": "initial query"},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "retry",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    res = query_rewriter_node(state)
    assert res["retry_count"] == 1
    assert "Ahmedabad" in res["subqueries"]["market"]

def test_compiled_graph_runnable():
    """Verify that the compiled LangGraph has valid structure and nodes."""
    graph = get_graph()
    assert graph is not None

# ==============================================================================
# Generalized City & Multi-Condition Tests
# ==============================================================================

def test_detect_city_multiple():
    """Verify city detection across various Indian cities."""
    from src.tools.web_search import detect_city
    assert detect_city("heart surgery cost in Mumbai") == "Mumbai"
    assert detect_city("cataract treatment in Delhi NCR") == "Delhi"
    assert detect_city("kidney stone operation in Ahmedabad") == "Ahmedabad"
    assert detect_city("knee replacement hospital in Bangalore") == "Bangalore"
    assert detect_city("hernia surgery in Pune hospitals") == "Pune"
    assert detect_city("what is the waiting period") is None

def test_router_classification_mumbai_cardiac():
    """Verify that a cardiac surgery query in Mumbai routes to agentic_hybrid."""
    state: AgentState = {
        "query": "How much does heart bypass surgery cost in Mumbai and what options exist?",
        "intent": "",
        "query_complexity": "",
        "subqueries": {},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = router_node(state)
    assert result["intent"] == "agentic_hybrid"

def test_router_classification_knee_replacement_delhi():
    """Verify that knee replacement in Delhi routes to agentic_hybrid."""
    state: AgentState = {
        "query": "Total knee replacement surgery cost and hospital options in Delhi?",
        "intent": "",
        "query_complexity": "",
        "subqueries": {},
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = router_node(state)
    assert result["intent"] == "agentic_hybrid"

def test_parallel_dual_retriever_cardiac_mumbai():
    """Verify parallel dual retriever retrieves both policy and market snippets for cardiac in Mumbai."""
    from src.graph.nodes import search_market_costs_node
    state: AgentState = {
        "query": "heart surgery cost in Mumbai",
        "intent": "agentic_hybrid",
        "query_complexity": "coverage_admissibility",
        "subqueries": {
            "procedure": "heart bypass surgery",
            "city": "Mumbai",
            "policy": "heart bypass surgery waiting period coverage",
            "market": "heart surgery cost in Mumbai"
        },
        "policy_chunks": [],
        "web_snippets": [],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    result = search_market_costs_node(state)
    assert len(result["policy_chunks"]) > 0
    assert len(result["web_snippets"]) > 0
    assert any("Mumbai" in s.get("title", "") or "Mumbai" in s.get("content", "") for s in result["web_snippets"])

def test_relevance_grader_generalized_cardiac():
    """Verify relevance grader passes for non-kidney cardiac surgery data."""
    state: AgentState = {
        "query": "heart bypass cost in Mumbai",
        "intent": "agentic_hybrid",
        "query_complexity": "coverage_admissibility",
        "subqueries": {"procedure": "heart surgery", "city": "Mumbai"},
        "policy_chunks": [],
        "web_snippets": [
            {
                "title": "Heart Surgery Cost in Mumbai",
                "content": "Angioplasty and CABG Bypass surgery in Mumbai costs between ₹1,50,000 and ₹5,00,000 at top cardiac centers.",
                "url": "https://example.com"
            }
        ],
        "grade_status": "",
        "retry_count": 0,
        "response": "",
        "execution_steps": []
    }
    res = relevance_grader_node(state)
    assert res["grade_status"] == "valid"

def test_search_healthcare_market_graceful_fallback():
    """Verify market search returns valid data structure even for rare procedures or unlisted cities."""
    res = search_healthcare_market(procedure="rare genetic therapy", city="Chandigarh")
    assert "procedure" in res
    assert "city" in res
    assert "snippets" in res
    assert isinstance(res["snippets"], list)
