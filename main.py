def main() -> None:
    print("AI Investigator 1.0. Ask me any question about the story")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("Not implemented yet. (Next step: RAG + LLM)\n")


if __name__ == "__main__":
    main()
