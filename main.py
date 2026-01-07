from investigator.story_loader import load_story_messages
from investigator.naive_rag import NaiveRAG
from investigator.llm_stub import StubLLM


def main() -> None:
    print("AI Investigator 1.0. Ask me any question about the story")

    messages = load_story_messages("whodunit_sydney.xml") 
    rag = NaiveRAG()
    rag.index(messages)

    llm = StubLLM()

    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print(rag.answer(q, llm))
        print()


if __name__ == "__main__":
    main()
