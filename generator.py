import os
from groq import Groq

GROQ_API_KEY = ""
MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

GREETINGS = {
    "hi", "hello", "hey", "hii", "hlo",
    "good morning", "good evening", "good afternoon"
}


def is_greeting(query):
    return query.lower().strip().rstrip("!.") in GREETINGS


def greeting_response(name="there", role="other"):
    role_lines = {
        "student": "Ask me about revenue, margins, growth — I'll explain the finance terms as we go.",
        "finance": "Ask me anything about the loaded reports — figures, ratios, trends.",
        "other": "Ask me anything about the loaded reports — I'll keep the jargon to a minimum."
    }
    return f"Hello {name}! I'm FinSight. {role_lines.get(role, role_lines['other'])}"


def generate_answer(query, retrieved_chunks, role="other"):

    if not retrieved_chunks:
        return "I don't have enough information to answer that."

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"""
SOURCE {i}
Company: {chunk['company']}
Page: {chunk['page']}

{chunk['text']}
"""
        )
    context = "\n\n-------------------\n\n".join(context_parts)

    style_instructions = {
        "student": (
            "Write for a finance student. After stating the number, briefly define "
            "any technical term used (e.g. 'Net Interest Margin (NIM) — the gap between interest earned and paid, as a % of assets'). "
            "Keep it educational but not long-winded."
        ),
        "finance": (
            "Write for a finance professional. Be direct and technical — lead with the number, "
            "use standard terminology without defining it, no hand-holding."
        ),
        "other": (
            "Write for someone with no finance background. Avoid jargon entirely. "
            "If a financial term must be used, explain it in plain, everyday language immediately after."
        )
    }
    style = style_instructions.get(role, style_instructions["other"])

    prompt = f"""You are a financial RAG assistant.

IMPORTANT RULES:
1. Use ONLY information present in the context.
2. Do NOT make assumptions.
3. Do NOT calculate metrics unless they are explicitly stated.
4. If the answer is not found, say: "I don't have enough information to answer that."
5. If a financial number is available, start your answer with the number.
6. End with: Source: Company Name, page X

STYLE (adapt your tone and explanation depth to this — this is important, follow it closely):
{style}

CONTEXT:
{context}

QUESTION:
{query}

Write a complete answer following the STYLE instructions above:"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()