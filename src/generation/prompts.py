# Prompt Templates

# Query Rewriter - Ultra strict, deterministic
QUERY_REWRITE_PROMPT = """Rewrite this query for document retrieval.

Rules:
- Keep the same meaning
- Make it specific and clear
- Max 30 words
- No additional questions
- No explanations
- Output ONLY the rewritten query

Query: {query}
Rewritten:"""


# Grounded Answer Generation
ANSWER_PROMPT = """Answer the question based ONLY on the provided context from the novel.

## Grounding Rules
1. Every claim must cite a source as [Chapter X]
2. If context is insufficient, say "I cannot answer this question based on the available context from the novel"
3. Do not speculate or add information not in the context
4. If multiple chapters are relevant, cite all of them

## Length Rules
- Default to 3-6 sentences
- Only provide longer explanations if the question explicitly asks for detail
- Be direct and concise

## Context from Novel
{context}

## Question
{question}

## Answer:"""


# Refusal template
REFUSAL_TEMPLATE = """I cannot answer this question based on the available context from the novel.

{reason}

Try asking:
- A more specific question about events or characters
- About a specific chapter or scene
- For details about a character mentioned in the novel"""


# Context formatting
def format_context(chunks: list) -> str:
    """Format retrieved chunks for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        chapter = chunk.get('chapter_title', f"Chapter {chunk.get('chapter_number', '?')}")
        content = chunk.get('content', '')
        parts.append(f"[Source {i} - {chapter}]\n{content}")
    return "\n\n---\n\n".join(parts)


PROMPTS = {
    'query_rewrite': QUERY_REWRITE_PROMPT,
    'answer': ANSWER_PROMPT,
    'refusal': REFUSAL_TEMPLATE,
}
