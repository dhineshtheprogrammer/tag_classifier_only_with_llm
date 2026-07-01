from .CustomAOI_helper import InitAOI

llmModels = ["gpt-35-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5.1"]
deploymentNames = ["genAICoEDevelopmentAndTesting",
                   "genAICoEDevelopmentAndTestingGPT4oLLM",
                   "genAICoEDevelopmentAndTesting4oMini",
                   "genAICoEDevelopmentAndTestingGPT5v1"]
embModel = "text-embedding-ada-002"
embDeploymentName = "DevelopmentAndTestingEmbedder"

secure_models = InitAOI(__name__, "USERNAME", "PASSWORD", llmModels, deploymentNames, embModel, embDeploymentName)


############ HOW TO USE THE CODE ###################
# Demo only — guarded so importing this module (as pipeline.py/classify.py do)
# never runs it. Run directly with `python -m src.helper` to try it.

if __name__ == "__main__":
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.llms import ChatMessage, MessageRole

    # Load documents and create a vector index using the default LLM
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine(temperature=0)
    print("Strict (temperature=0):")
    response = query_engine.query("Please give me details about John Wick.")
    print(str(response))

    query_engine = index.as_query_engine(temperature=1)
    print("\nOpen-ended (temperature=1):")
    response = query_engine.query("Please summarize this document on John Doe.")
    print(str(response))

    # Example of a chat completion using the default (active) LLM
    chat_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Write a short poem about technology."),
    ]
    response = secure_models.chat(chat_messages)
    print("Chat completion response:")
    print(response.message.content)

    # Example of using different LLM models directly without changing the default
    print("\n--- Using Different LLM Models Directly ---")
    for idx in range(secure_models.get_llm_count()):
        response = secure_models.chat(chat_messages, llm_index=idx)
        print(f"{llmModels[idx]} response:")
        print(response.message.content)
        print("\n" + "-" * 50 + "\n")

    # Switch back to the default LLM for general use
    secure_models.set_active_llm(0)

    # Example of text completion across different models
    prompt = "Complete this sentence: The future of artificial intelligence is"
    print("Text completion with different models:")
    for idx in range(secure_models.get_llm_count()):
        completion = secure_models.complete(prompt, llm_index=idx)
        print(f"Model {idx} ({llmModels[idx]}):")
        print(f"{prompt} {completion.text}")
        print("-" * 30)

    # Example of generating embeddings
    texts = [
        "Artificial intelligence is transforming industries",
        "Machine learning models require large amounts of data",
        "Neural networks are inspired by the human brain",
    ]
    embeddings = secure_models.embedder.get_text_embedding_batch(texts)
    print("\nGenerated embeddings:")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of each embedding: {len(embeddings[0])}")

    # Example of reading back per-call evaluation metrics
    metrics = secure_models._metrics.get_metrics()
    print(f"Total evaluations recorded: {len(metrics)}")
    print("\n" + "=" * 50)
    print("METRICS: EVALUATION (ALL EVALUATED CALLS)")
    print("=" * 50 + "\n")

    first_request_id = list(metrics.keys())[0]
    first_metrics = metrics[first_request_id]
    print(f"Sample metrics for request {first_request_id}:")
    print(f"  Username: {first_metrics.get('username')}")
    print(f"  Timestamp: {first_metrics.get('timestamp')}")
    print(f"  Latency: {first_metrics.get('metrics', {}).get('latency', 'N/A')} seconds")

    print("\nAll metrics for this request:")
    for key, value in first_metrics.get("metrics", {}).items():
        print(f"  {key}: {value}")
    print("\n" + "-" * 50 + "\n")
