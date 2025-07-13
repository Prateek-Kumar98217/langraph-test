from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
# === Init ===
llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.2)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(["your name is Yomun, a memory manager"], embedding=embedding_model)

# === State ===
class MemoryState(TypedDict):
    #add more state attributes for managing internal states and partial results between nodes
    messages: Annotated[list[BaseMessage], add_messages]
    relevant_memory: list[str]
    structured_memory: str

# === Memory Retriever ===
class MemoryRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def __call__(self, state: MemoryState):
        query = state["messages"][-1].content
        results = self.vectorstore.similarity_search(query, k=3)
        memory = [r.page_content for r in results]
        print(f"[MemoryRetriever] Memory for '{query}':", memory)
        return {"relevant_memory": memory}

# === Memory Evaluator ===
class MemoryEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: MemoryState):
        user_input = state["messages"][-1].content
        eval_prompt = (
            f"The user said: '{user_input}'.\n"
            "Does this contain factual or personal info worth remembering?\n"
            "Respond with only 'Yes' or 'No'."
        )
        decision = self.llm.invoke(eval_prompt).content.strip().lower()
        print(f"[MemoryEvaluator] Decision for '{user_input}':", decision)
        return {"store": "yes" in decision}

# === Memory Creator ===
class MemoryCreator:
    def __call__(self, state: MemoryState):
        user_input = state["messages"][-1].content
        # Can later use llm or hard coded memory structuring for better memory metadata and retrival
        structured = user_input.strip()
        print("[MemoryCreator] Created:", structured)
        return {"structured_memory": structured}

# === Memory Updater ===
class MemoryUpdater:
    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model=embedding_model

    def __call__(self, state: dict):
        memory = state.get("structured_memory", "")
        if memory:
            self.vectorstore.add_texts([memory], embedding=self.embedding_model)
            print("[MemoryUpdater] Stored:", memory)
        #for testing vectorstore updates
        #print("[MemoryUpdater] Current memory in store:")
        #print(self.vectorstore.similarity_search("SHOW_ALL", k=10)) 

        return {}

# === Build Subgraph ===
def create_memory_subgraph():
    sg = StateGraph(MemoryState)
    sg.add_node("retriever", MemoryRetriever(vectorstore))
    sg.add_node("evaluator", MemoryEvaluator(llm))
    sg.add_node("creator", MemoryCreator())
    sg.add_node("updater", MemoryUpdater(vectorstore, embedding_model))

    sg.add_edge(START, "retriever")
    sg.add_edge("retriever", "evaluator")
    sg.add_conditional_edges("evaluator",lambda x: "creator" if x["store"] else "end", {"creator": "creator", "end": END}
)

    sg.add_edge("creator", "updater")
    sg.add_edge("updater", END)

    return sg.compile()

if __name__ == "__main__":
    memory_graph = create_memory_subgraph()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["q", "exit", "quit"]:
            break
        memory_graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "relevant_memory": [],
            "structured_memory": ""
        })
