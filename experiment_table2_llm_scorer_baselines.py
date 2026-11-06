import argparse
import textwrap


NECESSITY_PROMPT_TEMPLATE = textwrap.dedent(
    """
    SYSTEM PROMPT: TASK-CONDITIONED NECESSITY SCORING
    You are a strict scoring judge for accessibility-tree nodes, focusing ONLY on TASK-CONDITIONED NECESSITY.
    Assign ONE integer necessity score n in {0..10} per node.
    Return JSON only:
    {
      "results": [
        {
          "nodeId": "string",
          "necessity_score": 0-10 integer,
          "necessity_reason": ["short_reason", "..."],
          "confidence": 0.0-1.0 or null
        }
      ]
    }
    """
).strip()


SENSITIVITY_PROMPT_TEMPLATE = textwrap.dedent(
    """
    SYSTEM PROMPT: SENSITIVITY SCORING
    You are a strict scoring judge for accessibility-tree nodes, focusing ONLY on SENSITIVITY.
    Assign ONE integer sensitivity score s in {0..10} per node.
    Return JSON only:
    {
      "results": [
        {
          "nodeId": "string",
          "sensitivity_score": 0-10 integer,
          "sensitivity_reason": ["short_reason", "..."],
          "confidence": 0.0-1.0 or null
        }
      ]
    }
    """
).strip()


def run(args: argparse.Namespace) -> None:
    print("Step 1/2: table2 baseline placeholder")
    print("Skipped by design (user request): no LLM baseline execution in this release.")
    print("Step 2/2: done")
    if args.show_prompts:
        print("\n--- Necessity prompt template ---")
        print(NECESSITY_PROMPT_TEMPLATE)
        print("\n--- Sensitivity prompt template ---")
        print(SENSITIVITY_PROMPT_TEMPLATE)
    print("TODO: if needed later, plug Together AI client here using TOGETHER_API_KEY.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Placeholder for Table 2 LLM baselines.")
    parser.add_argument("--show_prompts", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
