from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import (
    router_node,
    policy_retriever_node,
    policy_qa_node,
    decompose_query_node,
    search_market_costs_node,
    relevance_grader_node,
    query_rewriter_node,
    agentic_synthesizer_node,
)

def route_intent_decision(state: AgentState) -> str:
    """Routes based on intent classified by router node."""
    if state.get("intent") == "agentic_hybrid":
        return "decomposer"
    return "policy_retriever"

def grade_evidence_decision(state: AgentState) -> str:
    """Routes based on relevance grader verdict."""
    if state.get("grade_status") == "retry":
        return "query_rewriter"
    return "agentic_synthesizer"

def build_health_insurance_graph() -> StateGraph:
    """
    Constructs and compiles the full LangGraph state machine.
    """
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("policy_retriever", policy_retriever_node)
    workflow.add_node("policy_qa", policy_qa_node)
    workflow.add_node("decomposer", decompose_query_node)
    workflow.add_node("market_search", search_market_costs_node)
    workflow.add_node("relevance_grader", relevance_grader_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("agentic_synthesizer", agentic_synthesizer_node)

    # Set Entry Point
    workflow.set_entry_point("router")

    # Add Conditional Routing from Router
    workflow.add_conditional_edges(
        "router",
        route_intent_decision,
        {
            "policy_retriever": "policy_retriever",
            "decomposer": "decomposer"
        }
    )

    # Standard Policy RAG Path
    workflow.add_edge("policy_retriever", "policy_qa")
    workflow.add_edge("policy_qa", END)

    # Agentic Hybrid Path
    workflow.add_edge("decomposer", "market_search")
    workflow.add_edge("market_search", "relevance_grader")
    workflow.add_conditional_edges(
        "relevance_grader",
        grade_evidence_decision,
        {
            "query_rewriter": "query_rewriter",
            "agentic_synthesizer": "agentic_synthesizer"
        }
    )
    workflow.add_edge("query_rewriter", "market_search")
    workflow.add_edge("agentic_synthesizer", END)

    return workflow.compile()

# Cached compiled graph instance
_app_graph = None

def get_graph():
    global _app_graph
    if _app_graph is None:
        _app_graph = build_health_insurance_graph()
    return _app_graph

def run_agent(query: str) -> AgentState:
    """
    Helper function to execute a query through the graph.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
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
    return graph.invoke(initial_state)
