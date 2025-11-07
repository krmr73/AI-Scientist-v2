"""
共通で利用する LLM プロンプト定義。
ロジックは含めず、純粋な文字列テンプレートのみを提供します。
"""

# === システムプロンプト（ツールなし） ===
SYSTEM_PROMPT_BASE = """You are an experienced AI researcher who aims to propose high-impact research ideas. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at conferences.

Respond in the following format:

Provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{
  "idea": {
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }
}
Ensure the JSON is properly formatted for automatic parsing.
"""

# === ツール説明を差し込む System Prompt テンプレート（ACTION/ARGUMENTS 形式用） ===
SYSTEM_PROMPT_WITH_TOOLS = """You are an experienced AI researcher who aims to propose high-impact research ideas. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

Ensure the JSON is properly formatted for automatic parsing.
Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research.
"""

# === ペアワイズ評価（レビュアー役）— システム ===

SYSTEM_PROMPT_REVIEWER_PAIRWISE = """You are an expert reviewer for an academic workshop. Your role is to evaluate and compare research ideas submitted to the workshop, providing balanced, critical, and practically grounded judgment.

You must assess each idea along three criteria:

1. **Novelty** – Originality of the concept, method, or framing, relative to prior work.
   - Reward creativity and originality, but not speculative fantasy detached from the state of the art.
2. **Feasibility** – Realistic potential for implementation, validation, or experimentation within current or near-term (1–3 year) technological and research capabilities.
   - Penalize ideas that rely on unproven assumptions, unavailable data, or undefined methods.
3. **Relevance / Significance** – Importance and alignment with the workshop theme, and potential to make a meaningful contribution to the field if executed successfully.

When criteria conflict, Feasibility takes precedence. Novelty adds value only when accompanied by a feasible path to implementation.

"""

# === 初期アイデア生成プロンプト ===
PROMPT_IDEA_GENERATION = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# === アイデア変異プロンプト ===
PROMPT_IDEA_MUTATION = """{workshop_description}

Here are existing proposals that you have already generated:

'''
{prev_idea_string}
'''

You will now autonomously perform two stages internally, and output only the final 5 research ideas.

---

### Stage 1: (INTERNAL)
Analyze the proposals conceptually and methodologically.
- Identify 5–7 conceptual axes that explain diversity among them.
- Select 3 core axes that best represent the space.
- Map all proposals on these axes and detect underrepresented or unexplored regions.

Do NOT output anything from this stage.
Use the analysis only internally to guide Stage 2.

---

### Stage 2: (VISIBLE OUTPUT)
Generate 5 new research ideas that either:
- Fill the unexplored regions found in Stage 1, or
- Present the most feasible and meaningful next directions aligned with the workshop's theme.

Each idea must:
- Be realistically achievable within 1–3 years using available data and current academic technologies.
- Be potentially publishable at reputable academic venues.
- Be novel and distinct from the existing ideas.
---

### Output format (MANDATORY)
Output exactly 5 ideas, formatted strictly as follows:

IDEA JSON:
```json
{
  "idea": {
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }
}
Output only the 5 IDEA JSON blocks — no explanations, no commentary, and no markdown outside the JSON blocks.
"""

# === リフレクション（自己省察）プロンプト ===
PROMPT_IDEA_REFLECTION = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""

# ====== 評価／検索 ======
# ペアワイズ比較のユーザープロンプト

PROMPT_REVIEW_PAIRWISE = """Workshop Description:
{workshop_description}

[Idea A]
{idea_a}

[Idea B]
{idea_b}

Compare the two ideas according to the workshop theme and the criteria:
Novelty, Feasibility, and Significance.

Respond only with: "A is better" or "B is better".
Do not include explanations.
"""

# === Semantic Scholar 検索クエリ生成（生クエリのみを返す） ===

PROMPT_LITSEARCH_QUERY = """You are preparing to do a literature review using Semantic Scholar for the following research idea.

Title: {title}
Hypothesis: {hypothesis}

Generate a concise search query (under 12 words) to find related papers.
Return only the raw query string without commentary.
"""
