# Prompt Templates - Optimized for Qwen3

# Query Rewriter - Ultra strict, deterministic
# /no_think disables Qwen3's thinking mode for clean output
QUERY_REWRITE_PROMPT = """/no_think
Rewrite this query for document retrieval. Output ONLY the rewritten query, nothing else.

Query: {query}
Rewritten:"""


# Grounded Answer Generation
# /no_think ensures clean output without thinking tags
ANSWER_PROMPT = """/no_think
You are a literary expert answering questions about a novel. Answer based ONLY on the provided context.

RULES:
1. Use ONLY information from the context below
2. Cite sources as [Chapter X] for each claim
3. If the context doesn't contain the answer, say "I cannot find this information in the novel"
4. Be concise: 2-4 sentences for simple questions, more for complex ones
5. Never invent or assume information not in the context

CONTEXT FROM NOVEL:
{context}

QUESTION: {question}

ANSWER:"""


# Refusal template - cleaner, less verbose
REFUSAL_TEMPLATE = """I cannot answer this question based on the available context.

Reason: {reason}

Try asking about specific characters, events, or scenes from the novel."""


# Reranker prompt - optimized for JSON output
# /no_think is critical here to get clean JSON
RERANK_PROMPT = """/no_think
Rate how well this passage answers the query. Return ONLY valid JSON.

Query: {query}

Passage: {passage}

Scoring:
- 0-2: Not relevant
- 3-5: Somewhat related  
- 6-8: Directly relevant
- 9-10: Perfect answer

Output format: {{"score": <number>, "reason": "<brief reason>"}}

JSON:"""


# Context formatting - improved readability
def format_context(chunks: list) -> str:
    """Format retrieved chunks for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        chapter_num = chunk.get('chapter_number', '?')
        chapter_title = chunk.get('chapter_title', '')
        content = chunk.get('content', '')
        
        # Clean format with clear boundaries
        header = f"[Chapter {chapter_num}"
        if chapter_title:
            header += f": {chapter_title}"
        header += "]"
        
        parts.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(parts)


PROMPTS = {
    'query_rewrite': QUERY_REWRITE_PROMPT,
    'answer': ANSWER_PROMPT,
    'refusal': REFUSAL_TEMPLATE,
    'rerank': RERANK_PROMPT,
}
