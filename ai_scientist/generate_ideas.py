import argparse
import json
import os
import os.path as osp
import re
import sys
import traceback
from typing import Any, Dict, List

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import AVAILABLE_LLMS, create_client, get_response_from_llm
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
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Now, propose an idea that is {focus_statement}.
{focus_bullets}

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""


idea_focus_point_prompt = {
    "Novelty": {
        "focus_statement": "**highly novel and original**",
        "focus_bullets": """- Aim for a bold and imaginative approach, pushing boundaries or taking creative risks.
- Emphasize how this approach explores an unconventional angle and breaks new ground.
- The idea should still be implementable within constraints, but feel free to propose methods beyond typical incremental tweaks.""",
        "thought_instructions": """In <THOUGHT>, first discuss your intuitions and motivations for the idea.
Detail your high-level plan, necessary design choices, and ideal outcomes of the experiments.
Justify how the idea is **novel** relative to existing concepts and approaches.""",
    },
    "Feasibility": {
        "focus_statement": "**highly feasible and practical**",
        "focus_bullets": """- The approach should be straightforward to execute using current methods, minimizing complexities.
- Aim for a solution that is likely to yield reliable and useful results within the existing constraints.
- Clearly specify why this plan can be realistically achieved with minimal risk.""",
        "thought_instructions": """In <THOUGHT>, discuss the practical motivations and realistic plan for the idea.
Demonstrate how each step is achievable with existing code, time, or resource constraints.""",
    },
    "Interestingness": {
        "focus_statement": "**particularly intriguing or thought-provoking**",
        "focus_bullets": """- Focus on what makes the idea intellectually compelling or surprising.
- The approach may be moderate in complexity, but should offer insights that spark further questions or open new perspectives.
- Emphasize how the results could lead to fascinating discussions or follow-up experiments.""",
        "thought_instructions": """In <THOUGHT>, discuss why the idea is intriguing and how it can uncover unique insights about the urn model
or lead to deeper understanding.""",
    },
}


def generate_initial_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    criteria: str,
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

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)
            msg_history = []

            # Use the initial idea generation prompt
            prompt_text = idea_generation_prompt.format(
                workshop_description=workshop_description,
                prev_ideas_string=prev_ideas_string,
                focus_statement=idea_focus_point_prompt[criteria]["focus_statement"],
                focus_bullets=idea_focus_point_prompt[criteria]["focus_bullets"],
            )

            response_text, msg_history = get_response_from_llm(
                prompt=prompt_text,
                client=client,
                model=model,
                system_message=system_prompt,
                msg_history=msg_history,
            )

            arguments_text = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL).group(1)

            # Parse arguments
            try:
                arguments_json = json.loads(arguments_text)
                idea = arguments_json.get("idea")
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


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 5
    CRITERIAS = ["Interestingness", "Novelty", "Feasibility"]

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
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    idea_rname = args.workshop_file.replace(".md", "")
    print(idea_rname)
    os.makedirs(idea_rname, exist_ok=True)
    print("Starting idea generation for", idea_rname)

    for cri in CRITERIAS:
        ideas = generate_initial_idea(
            idea_fname=f"{idea_rname}/{cri}.json",
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            criteria=cri,
            max_num_generations=args.max_num_generations,
        )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
