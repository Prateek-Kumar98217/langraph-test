import json
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

@tool
def add_two_numbers(number1:float, number2:float)->float:
    """Adds two numbers"""
    return number1+number2

@tool
def multiply_two_numbers(number1: float, number2: float)->float:
    """multipies two numbers"""
    return number1*number2

@tool
def fallback_message():
    """fallback message when tool execution fails"""
    return "Sorry could not complete the requested action"

llm=init_chat_model("google_genai:gemini-2.0-flash", temperature=0.2)
llm_with_tools=llm.bind_tools([add_two_numbers, multiply_two_numbers, fallback_message])

class State(TypedDict):
    messages:Annotated[list, add_messages]

graph_builder=StateGraph(State)

def chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
    def __init__(self, tools:list)->None:
        self.tools_by_name={tool.name: tool for tool in tools}
    def __call__(self, inputs: dict):
        if messages:=inputs.get("messages", []):
            message=messages[-1]
        else:
            raise ValueError("There were no input message in the state")
        outputs=[]
        for tool_call in message.tool_calls:
            tool_name=tool_call["name"]
            tool_id=tool_call["id"]
            tool=self.tools_by_name.get(tool_name)
            if not tool:
               print(f"Tool by the name of {tool_name} not found in avaliable tools. Skipping...")
               continue
            try:
               print(f"Calling the tool {tool_name}...")
               result=tool.invoke(tool_call["args"])
            except Exception as e:
                print(f"First attempt failed: {e}")
                try:
                    print("Retrying tool call")
                    result=tool.invoke(tool_call["args"])
                except Exception as e2:
                    print(f"Failed again: {e2}")
                    fallback=self.tools_by_name.get("fallback_message")
                    result=fallback.invoke({}) if fallback else "Tool failed"
            outputs.append(
                ToolMessage(
                    content=result,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            )
        return {"messages": outputs}
               
            
    
graph_builder.add_node("tools", BasicToolNode(tools=[add_two_numbers, multiply_two_numbers, fallback_message]))

def route_tools(state: State):
    if isinstance(state, list):
        ai_message=state[-1]
    elif messages:=state.get("messages", []):
        ai_message=messages[-1]
    else:
        raise ValueError("No input message recieved in the tool edge")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls)>0:
        return "tools"
    return END

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph=graph_builder.compile()

def stream_agent(user_input: str):
    for events in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in events.values():
            msgs=value["messages"]
            if isinstance(msgs, list):
                print("Assistant: ", msgs[-1].content)
            else:
                print("Assistant: ", msgs.content)

while True:
    try:
        user_input=input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
        stream_agent(user_input)
    except:
        user_input="add 30 and 40"
        print("User:" + user_input)
        stream_agent(user_input)
        break
