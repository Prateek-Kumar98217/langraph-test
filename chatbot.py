from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

llm=init_chat_model("google_genai:gemini-2.0-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder=StateGraph(State)

def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph=graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages":[{"role":"user", "content":user_input}] }):
        for value in event.values():
            print("Assistant: ", value["messages"].content)

while True:
    try:
        user_input=input("User: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye")
            break
        stream_graph_updates(user_input)
    except:
        user_input="what is the secrret of life?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break