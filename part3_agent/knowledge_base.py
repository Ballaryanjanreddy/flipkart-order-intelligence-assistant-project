"""
Task 1 -- Policy knowledge base.

14 short Flipkart-style policy documents (2-4 sentences each), covering:
return windows by category, COD refund timelines, delivery SLAs, and
reverse-pickup eligibility (the brief's minimum-coverage list), plus a few
extra documents for realistic breadth.

Chunking strategy: SENTENCE-WISE, not fixed-size or overlapping-window.
Each document's sentences become individual chunks, and every chunk keeps a
`doc_id` pointer back to its parent document -- this is required because
Task 10's retrieval evaluation (Precision@3 / Recall@3) is scored at the
DOCUMENT level, not the chunk level, so multiple retrieved chunks from the
same document must be de-duplicated back to one document hit.
"""

import re

DOCUMENTS = [
    {
        "doc_id": "D01",
        "title": "Apparel & Footwear Return Window",
        "text": (
            "Apparel and footwear items purchased on Flipkart can be returned "
            "within 30 days of delivery. The item must be unused, unwashed, and "
            "returned with all original tags and packaging intact. Innerwear, "
            "swimwear, and customized apparel are not eligible for return."
        ),
    },
    {
        "doc_id": "D02",
        "title": "Electronics Return Window",
        "text": (
            "Most electronics items have a 10-day return window from the date "
            "of delivery. Mobile phones and laptops must be returned in their "
            "original box with all accessories, manuals, and invoice included. "
            "Physical damage or missing accessories will result in return rejection."
        ),
    },
    {
        "doc_id": "D03",
        "title": "Home & Furniture Return Window",
        "text": (
            "Home and furniture products can be returned within 15 days of "
            "delivery if they are defective, damaged, or significantly different "
            "from what was ordered. Assembled furniture may be subject to a "
            "inspection fee if the assembly hardware is missing on return."
        ),
    },
    {
        "doc_id": "D04",
        "title": "COD Refund Timeline",
        "text": (
            "For Cash on Delivery orders, refunds are issued to the customer's "
            "bank account or Flipkart wallet within 7-10 business days after the "
            "returned item passes quality inspection at the warehouse. Customers "
            "must provide valid bank account details during the return request "
            "since no card was used at purchase."
        ),
    },
    {
        "doc_id": "D05",
        "title": "Prepaid Refund Timeline",
        "text": (
            "For prepaid orders (card, UPI, or wallet), refunds are processed to "
            "the original payment method within 3-5 business days of the return "
            "being approved. UPI refunds are typically the fastest, often "
            "completing within 24-48 hours of approval."
        ),
    },
    {
        "doc_id": "D06",
        "title": "Standard Delivery SLA",
        "text": (
            "Standard delivery timelines range from 2 to 7 business days "
            "depending on the delivery pincode and product category. Metro "
            "cities generally receive orders within 2-4 days, while remote "
            "pincodes may take up to 7 days."
        ),
    },
    {
        "doc_id": "D07",
        "title": "Express Delivery SLA",
        "text": (
            "Express delivery, where available, guarantees next-day delivery "
            "for orders placed before 6 PM local time. Express delivery is only "
            "offered in select metro and tier-1 city pincodes and carries an "
            "additional shipping fee."
        ),
    },
    {
        "doc_id": "D08",
        "title": "Reverse Pickup Eligibility",
        "text": (
            "Reverse pickup (Flipkart arranging courier collection of a "
            "returned item) is available in most serviceable pincodes for "
            "apparel, footwear, electronics, and home categories. In pincodes "
            "without reverse pickup service, customers must self-ship the item "
            "using a provided prepaid shipping label."
        ),
    },
    {
        "doc_id": "D09",
        "title": "Damaged or Defective Item Policy",
        "text": (
            "If an item arrives damaged, defective, or significantly not as "
            "described, customers can request a replacement or full refund "
            "regardless of the category's normal return window. Photo or video "
            "evidence of the damage is required within 48 hours of delivery."
        ),
    },
    {
        "doc_id": "D10",
        "title": "Cancellation Before Shipping",
        "text": (
            "Orders can be cancelled free of charge any time before they are "
            "shipped from the warehouse. Once an order status changes to "
            "'Shipped', cancellation is no longer possible and the customer "
            "must use the return process instead after delivery."
        ),
    },
    {
        "doc_id": "D11",
        "title": "Exchange Policy",
        "text": (
            "Apparel and footwear items are eligible for a one-time free size "
            "or color exchange within the same 30-day return window, subject to "
            "stock availability. Exchanged items ship out only after the "
            "original item is picked up and passes quality inspection."
        ),
    },
    {
        "doc_id": "D12",
        "title": "Beauty & Personal Care Return Policy",
        "text": (
            "Beauty and personal care items are non-returnable once opened or "
            "used, due to hygiene regulations. Unopened, sealed beauty products "
            "can be returned within 10 days of delivery if the seal is intact."
        ),
    },
    {
        "doc_id": "D13",
        "title": "Order Tracking & Delays",
        "text": (
            "Customers can track real-time order status from the 'My Orders' "
            "section of the app or website. If a delivery is delayed beyond the "
            "estimated SLA by more than 2 days, customers are eligible to "
            "request expedited resolution or a partial shipping refund."
        ),
    },
    {
        "doc_id": "D14",
        "title": "High-Value Item Verification",
        "text": (
            "Orders above 20,000 rupees may require an OTP-based identity "
            "verification at the time of delivery for security purposes. "
            "Customers should ensure their registered mobile number is "
            "reachable at the time of delivery for high-value electronics "
            "and jewelry orders."
        ),
    },
]


def sentence_split(text: str):
    """Simple sentence splitter: splits on '. ' boundaries while keeping the
    period, good enough for our short, cleanly-punctuated policy documents."""
    text = text.strip()
    sentences = re.split(r"(?<=[.])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def build_chunks():
    """Returns a flat list of chunk dicts, each with a pointer back to its
    parent document (doc_id, title) -- this mapping is what Task 10's
    document-level scoring relies on."""
    chunks = []
    for doc in DOCUMENTS:
        sentences = sentence_split(doc["text"])
        for i, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}-c{i}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "text": sentence,
            })
    return chunks


# --- Task 1's retrieval-evaluation answer key (used by Task 10) ---
# For each realistic test query, the doc_id(s) a human would consider
# genuinely relevant. Used to compute Precision@3 / Recall@3 in
# evaluate_retrieval.py.
RETRIEVAL_ANSWER_KEY = [
    {
        "query": "How long do I have to return a t-shirt I bought?",
        "relevant_doc_ids": ["D01"],
    },
    {
        "query": "My laptop order came with a scratched screen, what can I do?",
        "relevant_doc_ids": ["D09", "D02"],
    },
    {
        "query": "When will I get my refund if I paid cash on delivery?",
        "relevant_doc_ids": ["D04"],
    },
    {
        "query": "Is there a pickup service for returning shoes, or do I have to ship them myself?",
        "relevant_doc_ids": ["D08"],
    },
    {
        "query": "Can I cancel my order after it has already shipped?",
        "relevant_doc_ids": ["D10"],
    },
    {
        "query": "How fast is delivery to a metro city?",
        "relevant_doc_ids": ["D06"],
    },
    {
        "query": "Can I return an opened bottle of shampoo?",
        "relevant_doc_ids": ["D12"],
    },
]


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"{len(DOCUMENTS)} documents -> {len(chunks)} sentence-wise chunks")
    for c in chunks[:5]:
        print(c)
    print(f"\n{len(RETRIEVAL_ANSWER_KEY)} retrieval-evaluation queries defined.")
