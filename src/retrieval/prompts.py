"""Single source of truth for all LLM prompts in the system.

Each prompt is versioned with a ``_V{N}`` suffix so evaluation runs can be
compared against specific versions.  The unversioned aliases (e.g.
``SYSTEM_PROMPT``) always point to the current production version and are
what the rest of the codebase should import.

Versioning convention:
    - Increment V number when the prompt text changes in a way that may affect
      model output (wording, structure, rules added/removed).
    - Keep old versions in this file so experiment results remain reproducible.
    - Update the active-version alias at the bottom of each section.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# V1 — initial version
SYSTEM_PROMPT_V1 = """\
You are a legal information assistant specialized in the Mexican Penal Code \
(Código Penal Federal). You answer questions based strictly on the provided \
context.

Rules:
1. Only use information from the provided context to answer.
2. Always cite specific article numbers (e.g., "Artículo 123") when \
referencing the law.
3. If the context does not contain enough information to answer, say so \
explicitly.
4. You are NOT providing legal advice. You are providing information about \
what the law states.
5. Respond in the same language as the question (Spanish or English).
"""

# Active version
SYSTEM_PROMPT = SYSTEM_PROMPT_V1

# ---------------------------------------------------------------------------
# Query answer template
# ---------------------------------------------------------------------------

# V1 — initial version
QUERY_TEMPLATE_V1 = """\
Based on the following excerpts from the Mexican Penal Code, answer the \
user's question.

## Context
{context}

## Question
{question}

## Instructions
- Cite specific article numbers in your answer.
- If the context is insufficient, state what information is missing.
- Be precise and factual.
"""

# Active version
QUERY_TEMPLATE = QUERY_TEMPLATE_V1

# ---------------------------------------------------------------------------
# Query expansion template
# ---------------------------------------------------------------------------

# V1 — initial version
EXPANSION_TEMPLATE_V1 = """\
Generate exactly 3 alternative phrasings of the following legal question in \
Spanish. Each phrasing should use different legal terminology to maximize \
search coverage. Return only the phrasings, one per line, no numbering.

Question: {question}
"""

# Active version
EXPANSION_TEMPLATE = EXPANSION_TEMPLATE_V1

# ---------------------------------------------------------------------------
# Convenience: registry for evaluation tooling
# ---------------------------------------------------------------------------

#: Maps (prompt_name, version) → template string.
#: Evaluation scripts can iterate this to test all historical versions.
PROMPT_REGISTRY: dict[tuple[str, str], str] = {
    ("system_prompt", "v1"): SYSTEM_PROMPT_V1,
    ("query_template", "v1"): QUERY_TEMPLATE_V1,
    ("expansion_template", "v1"): EXPANSION_TEMPLATE_V1,
}
