"""
Task 7 -- MOCK_LLM deterministic mode.

Zero network calls, zero API keys. Two responsibilities:
  1. classify_intent(): rule-based intent classification, with 2 few-shot
     examples baked into the matching logic (documented inline) as the
     brief requires -- these aren't just prompt text, they actively drive
     the routing decisions below.
  2. generate_response(): composes the final structured JSON answer
     (answer / source / confidence) from retrieved KB chunks and/or tool
     output, deterministically.

An optional USE_LIVE_LLM=1 path could call a real API here instead, but
MOCK_LLM is what every graded transcript must use, and nothing else in the
agent depends on live-LLM mode being present.
"""

import os
import re

USE_LIVE_LLM = os.environ.get("USE_LIVE_LLM") == "1"


# ----------------------------------------------------------------------------
# Intent classification
# ----------------------------------------------------------------------------
# Few-shot examples (also documented in the system prompt in agent_graph.py):
#   Example 1: "Will my order of a t-shirt be returned?" -> return_risk
#     (mentions an order + a risk/likelihood framing -> return_risk, not policy)
#   Example 2: "What category is this product photo?" -> product_category
#     (asks to classify/identify an image -> product_category)
# These two examples inform the keyword groups below rather than being inert
# prompt text -- they're why "will ... be returned" style phrasing routes to
# return_risk instead of policy_kb (a naive keyword match on "returned" alone
# would misroute this to the policy KB).

RETURN_RISK_PATTERNS = [
    r"\brisk\b", r"\blikely to (be )?return", r"\bwill .* be returned\b",
    r"\bchance of return", r"\breturn probability\b", r"\bpredict.*return\b",
]
IMAGE_PATTERNS = [
    r"\bimage\b", r"\bphoto\b", r"\bpicture\b", r"\bwhat category is this\b",
    r"\bclassify this\b", r"\.png\b",
]


def classify_intent(user_input: str) -> str:
    """Returns one of: 'policy', 'return_risk', 'product_category'."""
    lowered = user_input.lower()

    for pattern in RETURN_RISK_PATTERNS:
        if re.search(pattern, lowered):
            return "return_risk"

    for pattern in IMAGE_PATTERNS:
        if re.search(pattern, lowered):
            return "product_category"

    return "policy"  # default: assume a policy question


# ----------------------------------------------------------------------------
# Response generation
# ----------------------------------------------------------------------------

def generate_policy_response(retrieved_chunks: list, is_grounded: bool, top_similarity: float, threshold: float) -> dict:
    if not is_grounded:
        return {
            "answer": (
                f"I don't have a confident, grounded answer for that in our policy "
                f"knowledge base (best match similarity {top_similarity:.3f} is below "
                f"the {threshold:.2f} confidence threshold), so I won't guess. "
                f"Please contact a human support agent for this specific question."
            ),
            "source": "policy_kb",
            "confidence": round(top_similarity, 4),
        }

    # Deterministic composition: take the single best chunk verbatim as the
    # grounding fact, mention the source document.
    best = max(retrieved_chunks, key=lambda c: c["similarity"])
    answer = f"According to our {best['doc_title']} policy: {best['text']}"
    return {
        "answer": answer,
        "source": "policy_kb",
        "confidence": round(best["similarity"], 4),
    }


def generate_return_risk_response(tool_output: dict) -> dict:
    prob = tool_output["return_probability"]
    bucket = tool_output["risk_bucket"]
    answer = (
        f"This order has a predicted return probability of {prob:.1%}, "
        f"which falls in the '{bucket}' risk bucket "
        f"(threshold t*_rf={tool_output['t_star_rf']:.2f})."
    )
    return {
        "answer": answer,
        "source": "return_risk_tool",
        "confidence": prob,
    }


def generate_image_classification_response(tool_output: dict) -> dict:
    category = tool_output["predicted_category"]
    confidence = tool_output["confidence"]
    answer = f"This product image is classified as '{category}' with {confidence:.1%} confidence."
    return {
        "answer": answer,
        "source": "image_classifier_tool",
        "confidence": confidence,
    }


def generate_injection_deflection_response(matched_pattern: str) -> dict:
    return {
        "answer": (
            "I can't follow instructions embedded in a user message that try to "
            "override my behavior. I'm still happy to help with a genuine "
            "Flipkart order, policy, or product question."
        ),
        "source": "policy_kb",
        "confidence": 1.0,
    }
