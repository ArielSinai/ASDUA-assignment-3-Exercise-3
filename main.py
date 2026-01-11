from dotenv import load_dotenv
load_dotenv()

import argparse

from investigator.story_loader import load_story_messages
from investigator.llm_gemini import GeminiLLM
from investigator.naive_rag import NaiveRAG
from investigator.graph_rag import GraphRAG


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Investigator 1.0 – Ask questions about the story"
    )
    parser.add_argument(
        "--rag",
        choices=["naive", "graph"],
        default="naive",
        help="Which RAG system to use",
    )
    args = parser.parse_args()

    print("AI Investigator 1.0. Ask me any question about the story")

    # Load story
    messages = load_story_messages("whodunit_sydney.xml")

    # LLM (Gemini)
    llm = GeminiLLM()

    # Choose RAG system
    if args.rag == "naive":
        rag = NaiveRAG()
        rag.index(messages)
        answer_fn = lambda q: rag.answer(q, llm, top_k=8)

    elif args.rag == "graph":
        rag = GraphRAG()
        rag.index(messages)
        answer_fn = lambda q: rag.answer(q, llm)

    else:
        raise ValueError(f"Unknown RAG type: {args.rag}")

    # Interactive loop
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = answer_fn(question)
            print(answer)
        except Exception as e:
            print(f"ERROR: {e}")

        print()  # blank line between answers


if __name__ == "__main__":
    main()
