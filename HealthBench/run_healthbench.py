import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import common
from healthbench_eval import HealthBenchEval
from sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run HealthBench evals (healthbench, healthbench_hard) for one or more models."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=os.getenv("HEALTHBENCH_GRADER_MODEL", "gpt-5.2"),
        help="Model used to grade rubric items.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="healthbench,healthbench_hard",
        help="Comma-separated list: healthbench,healthbench_hard",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--examples", type=int)
    parser.add_argument("--n-threads", type=int, default=120)
    args = parser.parse_args()

    models = {
        "Baichuan-M2": ChatCompletionSampler(
            model="Baichuan-M2-32B",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            #max_tokens=2048,
            base_url=os.getenv("BAICHUAN_BASE_URL", "http://192.168.201.55:1217/v1"),
            api_key=os.getenv("BAICHUAN_API_KEY", "sk-your-secure-key-123"),
        ),
        "Ant-FP8": ChatCompletionSampler(
            model="Ant-FP8",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            #max_tokens=2048,
            base_url=os.getenv("BAICHUAN_BASE_URL", "http://192.168.201.55:1217/v1"),
            api_key=os.getenv("BAICHUAN_API_KEY", ""),
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if not args.model:
        raise ValueError("Missing required argument: --model")

    models_chosen = args.model.split(",")
    for model_name in models_chosen:
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Use --list-models.")
    models = {model_name: models[model_name] for model_name in models_chosen}

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_suffix = "_DEBUG" if args.debug else ""

    grading_sampler = ChatCompletionSampler(
        model=args.grader_model,
        system_message=OPENAI_SYSTEM_MESSAGE_API,
    )

    num_examples = args.examples if args.examples is not None else (5 if args.debug else None)

    evals_to_run = [e.strip() for e in args.eval.split(",") if e.strip()]
    for model_name, sampler in models.items():
        base_output_dir = Path("tmp") / model_name
        base_output_dir.mkdir(parents=True, exist_ok=True)

        evals: dict[str, HealthBenchEval] = {}
        for eval_name in evals_to_run:
            if eval_name == "healthbench":
                evals[eval_name] = HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                )
            elif eval_name == "healthbench_hard":
                evals[eval_name] = HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            else:
                raise ValueError(
                    f"Unknown --eval value: {eval_name}. Use healthbench or healthbench_hard."
                )

        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            file_stem = f"{eval_name}_{model_name}_{date_str}"

            report_filename = base_output_dir / f"{file_stem}{debug_suffix}.html"
            report_filename.write_text(common.make_report(result), encoding="utf-8")

            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            metrics = dict(sorted(metrics.items()))

            result_filename = base_output_dir / f"{file_stem}{debug_suffix}.json"
            result_filename.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            full_result_filename = (
                base_output_dir / f"{file_stem}{debug_suffix}_allresults.json"
            )
            result_dict = {
                "score": result.score,
                "metrics": result.metrics,
                "htmls": result.htmls,
                "convos": result.convos,
                "metadata": result.metadata,
            }
            full_result_filename.write_text(
                json.dumps(result_dict, indent=2),
                encoding="utf-8",
            )

            print(
                json.dumps(
                    {
                        "model": model_name,
                        "eval": eval_name,
                        "score": result.score,
                        "output_dir": str(base_output_dir),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
