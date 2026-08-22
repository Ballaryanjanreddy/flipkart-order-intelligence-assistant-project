"""
Tasks 5 & 6 -- LangGraph agent graph + prompt engineering.

Graph nodes (5 total, exceeds the brief's minimum of 4):
    1. guardrail_node   -- input-side prompt-injection check
    2. intent_node       -- policy / return_risk / product_category
    3. retrieval_node    -- RAG retrieval (policy path only)
    4. tool_node          -- calls check_return_risk or classify_product_image
    5. response_node     -- MOCK_LLM composes the final structured JSON answer

Conditional edges: guardrail_node branches on whether an injection was
detected; intent_node branches on the classified intent (policy vs.
return_risk vs. product_category). The graph does NOT run every node in
sequence every time -- e.g. a return_risk query skips retrieval_node
entirely, and a blocked-injection input skips intent/retrieval/tool nodes.

Conversational state: AgentState carries `conversation_memory` forward
between graph invocations for the SAME conversation (the caller is
responsible for passing the previous turn's returned state back in as the
next turn's input state -- this is what "short-term state, not built-in
chat memory" means). A freshly-started conversation simply starts from an
empty `conversation_memory` dict.

--------------------------------------------------------------------------
SYSTEM PROMPT -- annotated against the 4S principles + role prompting
--------------------------------------------------------------------------
(ROLE)     "You are Flipkart's customer support assistant."
(SPECIFIC) "Answer only using the retrieved policy chunk(s) provided, or
            the tool output provided. Never invent a policy, timeline, or
            number that isn't in the given context."
(SHORT)    "Keep answers to 1-3 sentences."
(SURROUND) "You will be given: retrieved_chunks (may be empty), tool_output
            (may be None), and the user's message. Use only what's given."
(SINGLE)   "Produce exactly one structured JSON object: {answer, source,
            confidence}. Never produce prose outside that JSON structure."

Few-shot intent-classification examples (also implemented as active
keyword-matching logic in mock_llm.classify_intent, not just prompt text):
    Example 1: "Will my order of a t-shirt likely be returned?"
               -> intent: return_risk
    Example 2: "What category is this product photo?"
               -> intent: product_category
--------------------------------------------------------------------------
"""

from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

from guardrails import check_prompt_injection, check_groundedness, GROUNDEDNESS_THRESHOLD
from mock_llm import (
    classify_intent,
    generate_policy_response,
    generate_return_risk_response,
    generate_image_classification_response,
    generate_injection_deflection_response,
)
from tools import check_return_risk, classify_product_image
from build_index import load_index_and_model, retrieve

SYSTEM_PROMPT = """You are Flipkart's customer support assistant.

(SPECIFIC) Answer only using the retrieved policy chunk(s) or tool output
you are given. Never invent a policy, timeline, or number that is not
present in the given context.

(SHORT) Keep answers to 1-3 sentences.

(SURROUND) You will be given: retrieved_chunks (may be empty), tool_output
(may be None), and the user's message. Use only what's given -- do not use
outside knowledge about Flipkart or e-commerce policy in general.

(SINGLE) Produce exactly one structured JSON object with fields: answer,
source (one of "policy_kb", "return_risk_tool", "image_classifier_tool"),
and confidence. Never produce free-form prose outside that JSON structure.

Few-shot intent examples:
  User: "Will my order of a t-shirt likely be returned?"
  -> intent: return_risk (asks about return LIKELIHOOD, not policy)

  User: "What category is this product photo?"
  -> intent: product_category (asks to classify an image)
"""


class AgentState(TypedDict):
    user_input: str
    conversation_memory: Dict[str, Any]
    is_injection: bool
    injection_pattern: Optional[str]
    intent: Optional[str]
    retrieved_chunks: List[dict]
    is_grounded: Optional[bool]
    top_similarity: Optional[float]
    tool_output: Optional[dict]
    response: Optional[dict]


# --- Lazily-loaded shared resources (index/model loaded once, not per call) ---
_INDEX = None
_CHUNKS = None
_EMBED_MODEL = None


def _get_index():
    global _INDEX, _CHUNKS, _EMBED_MODEL
    if _INDEX is None:
        _INDEX, _CHUNKS, _EMBED_MODEL = load_index_and_model()
    return _INDEX, _CHUNKS, _EMBED_MODEL


# ----------------------------------------------------------------------------
# Node 1: guardrail_node
# ----------------------------------------------------------------------------
def guardrail_node(state: AgentState) -> AgentState:
    result = check_prompt_injection(state["user_input"])
    state["is_injection"] = result["is_injection"]
    state["injection_pattern"] = result["matched_pattern"]
    return state


def route_after_guardrail(state: AgentState) -> str:
    return "blocked" if state["is_injection"] else "continue"


# ----------------------------------------------------------------------------
# Node 2: intent_node
# ----------------------------------------------------------------------------
def intent_node(state: AgentState) -> AgentState:
    state["intent"] = classify_intent(state["user_input"])
    return state


def route_after_intent(state: AgentState) -> str:
    return state["intent"]  # "policy" | "return_risk" | "product_category"


# ----------------------------------------------------------------------------
# Node 3: retrieval_node (policy path only)
# ----------------------------------------------------------------------------
def retrieval_node(state: AgentState) -> AgentState:
    index, chunks, model = _get_index()
    results = retrieve(state["user_input"], index, chunks, model, top_k=3)
    state["retrieved_chunks"] = results

    grounded = check_groundedness(results, threshold=GROUNDEDNESS_THRESHOLD)
    state["is_grounded"] = grounded["is_grounded"]
    state["top_similarity"] = grounded["top_similarity"]
    return state


# ----------------------------------------------------------------------------
# Node 4: tool_node (return_risk / product_category paths)
# ----------------------------------------------------------------------------
REFERENCE_PHRASES = ["same order", "that order", "the same", "instead"]


def tool_node(state: AgentState) -> AgentState:
    memory = state["conversation_memory"]
    lowered = state["user_input"].lower()

    if state["intent"] == "return_risk":
        refers_to_prior = any(p in lowered for p in REFERENCE_PHRASES)

        if refers_to_prior and "last_order_features" in memory:
            # Demonstrates state carried across turns: reuse prior order's
            # features, applying any simple override mentioned in this turn.
            order_features = dict(memory["last_order_features"])
            if "upi" in lowered:
                order_features["payment_method"] = "Prepaid_UPI"
            elif "cod" in lowered or "cash on delivery" in lowered:
                order_features["payment_method"] = "COD"
            elif "wallet" in lowered:
                order_features["payment_method"] = "Wallet"
            elif "card" in lowered:
                order_features["payment_method"] = "Prepaid_Card"
        elif refers_to_prior and "last_order_features" not in memory:
            # Demonstrates state correctly ABSENT in a fresh conversation:
            # no prior order to refer back to.
            state["tool_output"] = None
            state["response"] = {
                "answer": (
                    "I don't have a previous order in this conversation to "
                    "refer back to. Could you share the order's details "
                    "(category, price, payment method, etc.)?"
                ),
                "source": "return_risk_tool",
                "confidence": 0.0,
            }
            return state
        else:
            order_features = state.get("_pending_order_features")
            if order_features is None:
                state["tool_output"] = None
                state["response"] = {
                    "answer": "Please share the order's details so I can assess return risk.",
                    "source": "return_risk_tool",
                    "confidence": 0.0,
                }
                return state

        tool_output = check_return_risk(order_features)
        state["tool_output"] = tool_output
        memory["last_order_features"] = order_features  # persist for future turns
        memory["last_tool_output"] = tool_output

    elif state["intent"] == "product_category":
        image_path = state.get("_pending_image_path")
        if image_path is None:
            state["tool_output"] = None
            state["response"] = {
                "answer": "Please provide the path to a product image to classify.",
                "source": "image_classifier_tool",
                "confidence": 0.0,
            }
            return state
        tool_output = classify_product_image(image_path)
        state["tool_output"] = tool_output
        memory["last_image_classification"] = tool_output

    return state


# ----------------------------------------------------------------------------
# Node 5: response_node
# ----------------------------------------------------------------------------
def response_node(state: AgentState) -> AgentState:
    if state.get("response") is not None:
        return state  # already set by an earlier short-circuit (e.g. tool_node)

    if state["is_injection"]:
        state["response"] = generate_injection_deflection_response(state["injection_pattern"])
        return state

    if state["intent"] == "policy":
        state["response"] = generate_policy_response(
            state["retrieved_chunks"], state["is_grounded"],
            state["top_similarity"], GROUNDEDNESS_THRESHOLD,
        )
    elif state["intent"] == "return_risk":
        state["response"] = generate_return_risk_response(state["tool_output"])
    elif state["intent"] == "product_category":
        state["response"] = generate_image_classification_response(state["tool_output"])

    return state


# ----------------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("intent", intent_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail", route_after_guardrail,
        {"blocked": "response", "continue": "intent"},
    )

    graph.add_conditional_edges(
        "intent", route_after_intent,
        {
            "policy": "retrieval",
            "return_risk": "tool",
            "product_category": "tool",
        },
    )

    graph.add_edge("retrieval", "response")
    graph.add_edge("tool", "response")
    graph.add_edge("response", END)

    return graph.compile()


def run_turn(compiled_graph, user_input: str, conversation_memory: dict = None,
             pending_order_features: dict = None, pending_image_path: str = None) -> AgentState:
    """Runs one turn of the conversation. Pass the PREVIOUS turn's returned
    conversation_memory back in to carry state; pass None (or omit) to start
    a fresh conversation with no prior state."""
    initial_state: AgentState = {
        "user_input": user_input,
        "conversation_memory": conversation_memory if conversation_memory is not None else {},
        "is_injection": False,
        "injection_pattern": None,
        "intent": None,
        "retrieved_chunks": [],
        "is_grounded": None,
        "top_similarity": None,
        "tool_output": None,
        "response": None,
    }
    if pending_order_features is not None:
        initial_state["_pending_order_features"] = pending_order_features
    if pending_image_path is not None:
        initial_state["_pending_image_path"] = pending_image_path

    final_state = compiled_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    graph = build_graph()
    result = run_turn(graph, "How long do I have to return a t-shirt I bought?")
    print(result["response"])
