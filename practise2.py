#for better memory saving and retrieval use better prompts

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load keys
load_dotenv()

# Init LLM and embedding model
llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.2)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Init FAISS vectorstore with dummy text
vectorstore = FAISS.from_texts(["init"], embedding=embedding_model)

# Function to store to memory
def store_to_memory(text: str):
    vectorstore.add_texts([text])
    print(f"[Memory] Stored: '{text}'")

# Function to retrieve from memory
def retrieve_memory(query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    results = [doc.page_content for doc in docs if doc.page_content != "init"]
    print(f"[Memory] Retrieved for '{query}': {results}")
    return results

# Function to build prompt with context
def build_prompt(memory_context: list[str], user_message: str):
    memory_block = "\n".join(f"- {m}" for m in memory_context)
    system_prompt = f"Relevant memory:\n{memory_block}\n\n" if memory_context else ""
    full_input = system_prompt + user_message
    print(f"\n---- Prompt to LLM ----\n{full_input}\n------------------------\n")
    return full_input

# Main loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    # Step 1: Retrieve context
    recalled_memory = retrieve_memory(user_input)

    # Step 2: Build full prompt
    full_prompt = build_prompt(recalled_memory, user_input)

    # Step 3: Call LLM
    response = llm.invoke([HumanMessage(content=full_prompt)])
    print(f"Assistant: {response.content}")

    # Step 4: Decide whether to store
    decision_prompt = (
    f"The user said: '{user_input}'.\n"
    "Does this message contain factual or personal information (like names, preferences, locations, etc.) "
    "that should be remembered for future conversations?\n"
    "Reply only with 'Yes' or 'No'."
    )

    decision_output = llm.invoke(decision_prompt).content.strip()
    print(f"[Memory] Decision raw: '{decision_output}'")

    decision = decision_output.lower()
    if decision.startswith("yes"):
        store_to_memory(user_input)
    else:
        print("[Memory] Not stored.")

    if "yes" in decision:
        store_to_memory(user_input)
