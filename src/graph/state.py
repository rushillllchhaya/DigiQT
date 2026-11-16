from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Unified state schema for the Health Insurance Assistant.
    Tracks query, routing decisions, retrieval contexts, grading, and execution traces.
    """
    query: str
    intent: str  # "policy_qa" or "agentic_hybrid"
    query_complexity: str  # "simple_factual", "coverage_admissibility", "agentic_hybrid"
    subqueries: Dict[str, str]  # {"policy": str, "market": str, "city": str, "procedure": str}
    policy_chunks: List[Dict[str, Any]]
    web_snippets: List[Dict[str, Any]]
    grade_status: str  # "valid", "retry", "done"
    retry_count: int
    response: str
    execution_steps: List[Dict[str, Any]]
