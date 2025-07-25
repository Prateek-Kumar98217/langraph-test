import json
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch

load_dotenv()

tool = TavilySearch(max_results=2)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")

llm=init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools=llm.bind_tools(tools)

class State(TypedDict):
    messages:Annotated[list, add_messages]

graph_builder=StateGraph(State)

def chatbot(state:State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
    def __init__(self, tools:list)->None:
        self.tools_by_name={tool.name:tool for tool in tools}
    def __call__(self, inputs: dict):
        if messages:=inputs.get("messages", []):
            message=messages[-1]
        else:
            raise ValueError("No message recieved in input")
        outputs=[]
        for tool_call in message.tool_calls:
            tool_result=self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))
        return{"messages": outputs}
    
tool_node=BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    if isinstance(state, list):
        ai_message=state[-1]
    elif messages:=state.get("messages", []):
        ai_message=messages[-1]
    else:
        raise ValueError(f"No message found in input state to tool edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls)>0:
        return "tools"
    return END

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph=graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            msgs = value["messages"]
            if isinstance(msgs, list):
                print("Assistant:", msgs[-1].content)
            else:
                print("Assistant:", msgs.content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break