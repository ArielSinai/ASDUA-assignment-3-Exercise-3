# AI Investigator 1.0

AI Investigator 1.0 is a console-based Python program that lets a user ask free-text questions about a story that is told through structured message logs.

The goal of the project is to simulate a realistic investigation scenario where an LLM cannot receive the entire story at once. To do this, the system enforces a strict 3000-character prompt limit and relies on Retrieval-Augmented Generation (RAG) to select only the most relevant parts of the story for each question.

For every question, the system retrieves relevant messages, builds a prompt that stays within the allowed limit, and then asks the LLM (Gemini) to generate an answer based only on those messages. If the answer cannot be determined from the story, the system explains why instead of guessing.


# RAG Approaches Used

The project includes three different RAG implementations. All of them solve the same task, but each uses a different retrieval strategy. This makes it possible to compare their strengths and limitations under the same constraints.


# NaiveRAG – Embedding-Based Retrieval

This approach uses semantic embeddings to find messages that are similar in meaning to the user’s question. Each message is embedded, the question is embedded, and the closest matches are selected. Nearby messages can also be added to provide additional context.

This method works well for straightforward questions and questions that rely on semantic similarity, but it can struggle when the answer requires connecting information across multiple loosely related messages.


# GraphRAG – Graph-Based Multi-Hop Retrieval

GraphRAG treats the story as a lightweight graph. Messages are connected based on factors like time proximity and shared participants (sender or receiver).

The retrieval process starts by finding a few relevant “seed” messages using embeddings. From there, the system expands through the graph using multiple hops, allowing it to collect related messages that may not be semantically similar on their own.

This approach is especially useful for more complex questions where the answer is not stated directly in a single message, but instead must be inferred from several connected interactions.


# BM25RAG – Lexical (Keyword-Based) Retrieval

BM25RAG uses a classic information-retrieval technique based on keyword matching. Messages are ranked according to how much their text overlaps with the user’s question, without using embeddings or semantic models.

This method is fast, simple, and very transparent. It deliberately avoids inference and only works with what is explicitly written in the story. The LLM is still used to generate the final answer, but only after the relevant messages have been retrieved.

Because of this, BM25RAG will fail gracefully when a question relies on implied meaning rather than explicit wording.


