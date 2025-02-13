# def chat_loop(chain, retriever):
#     while True:
#         query = input("\nEnter your question (or 'exit' to quit): ").strip()
#         if query.lower() == 'exit':
#             break
#         if not query:
#             print("Please enter a valid question.")
#             continue
#         import time
#         start_time = time.time()
#         result = chain.invoke({
#             "context": retriever.invoke(query),
#             "input": query
#         })
#         elapsed = time.time() - start_time
#         if isinstance(result, dict) and "error" in result:
#             print(f"\nError: {result['error']}")
#             continue
#         print(f"\nAnswer (processed in {elapsed:.2f} seconds):")
#         print(result)
#         if isinstance(result, dict) and "sources" in result:
#             print("\nSources used:")
#             for i, source in enumerate(result["sources"], 1):
#                 print(f"\nSource {i}:")
#                 print(f"Content preview: {source['content']}...")
#                 print(f"Source: {source['metadata'].get('source', 'Unknown')}")
#                 print(f"Page: {source['metadata'].get('page', 'Unknown')}")
#                 if source.get('relevance_score'):
#                     print(f"Relevance Score: {source['relevance_score']:.2f}")
def chat_loop(chain, retriever):
    """
    Starts an interactive chat session.
    Type 'exit' to quit the interface.
    """
    print("Chat Interface Initialized. Type your query below (or 'exit' to quit).")
    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() == "exit":
                print("Exiting chat interface. Goodbye!")
                break
            if not query:
                print("Empty query detected. Please enter a valid question.")
                continue

            import time
            start_time = time.time()

            # Retrieve context and invoke the chain to generate an answer.
            retrieved_context = retriever.invoke(query)
            result = chain.invoke({
                "context": retrieved_context,
                "input": query
            })

            elapsed = time.time() - start_time

            # Check if result is a dictionary or a simple string.
            if isinstance(result, dict):
                answer = result.get("answer", result)
                sources = result.get("sources", [])
            else:
                answer = result
                sources = []

            print(f"\nAnswer (processed in {elapsed:.2f} seconds):")
            print(answer)

            if sources:
                print("\nSources used:")
                for idx, source in enumerate(sources, 1):
                    print(f"\nSource {idx}:")
                    content_preview = source.get("content", "No content available")[:200]
                    print(f"Preview: {content_preview}...")
                    metadata = source.get("metadata", {})
                    print(f"Source: {metadata.get('source', 'Unknown')}")
                    print(f"Page: {metadata.get('page', 'Unknown')}")
                    if source.get("relevance_score"):
                        print(f"Relevance Score: {source['relevance_score']:.2f}")
        except KeyboardInterrupt:
            print("\nChat session interrupted. Exiting.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
