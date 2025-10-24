import argparse
import json
import os
import os.path as osp
import sys
import traceback
from typing import Any, Dict, List

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    extract_json_block,
    get_idea_from_payload,
    get_response_from_llm,
)
from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]


system_prompt = f"""You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at conferences.

Respond in the following format:

Provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research."""

# Define the initial idea generation prompt
idea_generation_simple_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

idea_mutation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_idea_string}
'''

Now, modify one or more key aspects of the existing idea to create a novel yet feasible variation. Your new idea should retain core elements while introducing meaningful changes in assumptions, parameters, or methodology. Ensure that it remains impactful and investigable using the provided code.
Avoid trivial modifications or overly complex solutions that are impractical to implement.

If you have new information from literature search results. incorporate them into your reflection and modify your proposal accordingly.

Literature search results (if any):

{semantic_scholar_results}
"""


def generate_simple_initial_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for idx in range(max_num_generations):
        print()
        print(f"Generating proposal {idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)
            msg_history: List[Dict[str, str]] = []

            # Use the initial idea generation prompt
            prompt_text = idea_generation_simple_prompt.format(
                workshop_description=workshop_description,
                prev_ideas_string=prev_ideas_string,
            )

            response_text, msg_history = get_response_from_llm(
                prompt=prompt_text,
                client=client,
                model=model,
                system_message=system_prompt,
                msg_history=msg_history,
            )

            # Parse arguments
            try:
                payload = extract_json_block(response_text)  # 例: {"idea": {...}}
                idea = get_idea_from_payload(payload)  # "idea" の存在チェック込み

                idea["ID"] = f"0_{idx}"

                if not idea:
                    raise ValueError("Missing 'idea' in arguments.")

                # Append the idea to the archive
                idea_str_archive.append(json.dumps(idea))
                print(f"Proposal finalized: {idea}")
            except json.JSONDecodeError:
                raise ValueError("Invalid arguments JSON for FinalizeIdea.")

        except Exception:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


def generate_semantic_scholar_results(
    prev_idea: Dict[str, Any],
    client: Any,
    model: str,
    use_semantic_scholar: bool,
):

    if use_semantic_scholar:
        search_query_prompt = f"""
            You are preparing to do a literature review using Semantic Scholar for the following research idea.

            Title: {prev_idea.get('Title', '')}
            Hypothesis: {prev_idea.get('Short Hypothesis', '')}

            Generate a concise search query (under 12 words) to find related papers.
            Do not return commentary — only return the raw query string.
            """
        query, _ = get_response_from_llm(
            prompt=search_query_prompt,
            client=client,
            model=model,
            system_message="You are a helpful research assistant.",
            msg_history=[],
        )
        query = query.strip().strip('"')
        semantic_scholar_results = semantic_scholar_tool.use_tool(query=query)
    else:
        semantic_scholar_results = ""

    return semantic_scholar_results


def mutate_ideas(
    base_dir: str,
    client: Any,
    model: str,
    workshop_description: str,
    ideas: list,
    generation: int,
    use_semantic_scholar: bool = False,
) -> List[Dict]:

    idea_str_archive = []

    for i, prev_idea in enumerate(ideas):
        print()
        print(f"Generating proposal {i}/{len(ideas)}")
        try:
            prev_idea_string = json.dumps(prev_idea)
            msg_history = []

            semantic_scholar_results = generate_semantic_scholar_results(
                prev_idea=prev_idea,
                client=client,
                model=model,
                use_semantic_scholar=use_semantic_scholar,
            )

            # Use the initial idea generation prompt
            prompt_text = idea_mutation_prompt.format(
                workshop_description=workshop_description,
                prev_idea_string=prev_idea_string,
                semantic_scholar_results=semantic_scholar_results,
            )

            response_text, msg_history = get_response_from_llm(
                prompt=prompt_text,
                client=client,
                model=model,
                system_message=system_prompt,
                msg_history=msg_history,
            )

            # Parse arguments
            try:
                payload = extract_json_block(response_text)  # 例: {"idea": {...}}
                idea = get_idea_from_payload(payload)  # "idea" の存在チェック込み
                if not idea:
                    raise ValueError("Missing 'idea' in arguments.")

                idea["source_ids"] = [prev_idea["ID"]]
                idea["ID"] = f"{generation}_{i}"

                # # スコア評価
                # evaluation = evaluate_idea(idea, client, model)
                # idea.update(evaluation)

                # Append the idea to the archive
                idea_str_archive.append(json.dumps(idea))
                print(f"Proposal finalized: {generation}_{i}")
            except json.JSONDecodeError:
                raise ValueError("Invalid arguments JSON for FinalizeIdea.")

        except Exception:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    mutated_ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(f"{base_dir}/{generation}.json", "w") as f:
        json.dump(mutated_ideas, f, indent=4)
    return mutated_ideas


def pairwise_evaluate(
    idea_a: dict,
    idea_b: dict,
    client: Any,
    model: str,
    workshop_description: str,
):
    pairwise_evaluation_system_prompt = """
        You are an expert reviewer for an academic workshop. Your role is to evaluate and compare research ideas submitted to the workshop. You must provide an impartial and expert judgment based on the criteria of novelty, feasibility, and relevance to the workshop theme.
    """

    pairwise_evaluate_prompt = f"""
        Workshop Description:
        {workshop_description}

        Here are two research ideas:

        [Idea A]
        {idea_a}

        [Idea B]
        {idea_b}

        Which idea is better overall based on the workshop's theme and the following criteria:
        1. Novelty
        2. Feasibility
        3. Significance

        Please respond with only: "A is better" or "B is better".
        """.strip()

    try:
        response, _ = get_response_from_llm(
            prompt=pairwise_evaluate_prompt,
            client=client,
            model=model,
            system_message=pairwise_evaluation_system_prompt,
            msg_history=[],
        )
        response_text = response.strip().lower()

        if "a is better" in response_text:
            return idea_a
        elif "b is better" in response_text:
            return idea_b
        else:
            print("曖昧な応答:", response_text)
            return None

    except Exception as e:
        print("評価失敗:", e)
        return None


def evaluate_idea(idea: Dict[str, Any], client: Any, model: str) -> Dict[str, int]:
    """
    1つのアイデアに対して LLM にスコア付けをさせる。
    - client: LLMクライアント
    - model: モデル名（例: gpt-4o）
    戻り値は 3軸のスコア辞書
    """
    # 評価用の system prompt（役割指定）
    evaluation_system_prompt = (
        "You are a critical and objective reviewer for academic research proposals. "
        "Your task is to assign integer scores from 1 (very poor) to 20 (excellent) on the following dimensions:\n"
        "- Interestingness: Is the idea intellectually stimulating or surprising?\n"
        "- Novelty: Is it clearly original and different from prior work?\n"
        "- Feasibility: Can it be realistically implemented and tested in an academic lab?\n\n"
        "Return only a JSON block, without commentary or extra explanation."
    )

    # ユーザープロンプト（入力となるアイデア）
    eval_prompt = f"""
        Evaluate the following research proposal:

        {json.dumps(idea, indent=2)}

        Return your scores in this format:

        ```json
        {{
        "Interestingness": <int>,
        "Novelty": <int>,
        "Feasibility": <int>
        }}
        """.strip()

    try:
        response, _ = get_response_from_llm(
            prompt=eval_prompt, client=client, model=model, system_message=evaluation_system_prompt, msg_history=[]
        )

        score_json = extract_json_block(response)

        return {
            "Interestingness": int(score_json.get("Interestingness", 0)),
            "Novelty": int(score_json.get("Novelty", 0)),
            "Feasibility": int(score_json.get("Feasibility", 0)),
        }

    except Exception as e:
        print("評価失敗:", e)
        return {"Interestingness": 0, "Novelty": 0, "Feasibility": 0}


if __name__ == "__main__":
    CRITERIAS = ["Feasibility", "Interestingness", "Novelty"]

    parser = argparse.ArgumentParser(description="Generate AI scientist proposals - template free")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    idea_dir_name = args.workshop_file.replace(".md", "")
    os.makedirs(idea_dir_name, exist_ok=True)
    print("Starting idea generation for", idea_dir_name)

    ideas = generate_simple_initial_idea(
        idea_fname=f"{idea_dir_name}/initial_ideas.json",
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
