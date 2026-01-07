from dotenv import load_dotenv
load_dotenv()

from investigator.story_loader import load_story_messages
from investigator.naive_rag import NaiveRAG
from investigator.llm_gemini import GeminiLLM


def main() -> None:
    print("AI Investigator 1.0. Ask me any question about the story")

    messages = load_story_messages("whodunit_sydney.xml")
    rag = NaiveRAG()
    rag.index(messages)

    llm = GeminiLLM()

    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print(rag.answer(q, llm, top_k=5))
        print()


if __name__ == "__main__":
    main()
