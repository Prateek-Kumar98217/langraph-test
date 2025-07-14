from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage

load_dotenv()

llm=init_chat_model("google_genai:gemini-2.0-flash", temperature=0.2)

class ToolState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_name: str
    tool_input: str
    tool_output: str

class ToolSelector:
    def __init__(self, llm):
        self.llm=llm

    def __call__(self, state: ToolState):
        query=state["messages"][-1].content
        prompt=(
            f"the user said this: '{query}'.\n"
            "Which tool should be used? Choose from ['calculator', 'weather', 'none']. \n"
            "Reply with only the tool name"
        )
        tool=self.llm.invoke(prompt).content.strip().lower()
        print(f"[Tool Selector] Selected tool: {tool}")
        return{"tool_name": tool, "tool_input":query}
    
class Calculator:
    def __call__(self, state: ToolState):
        query=state["tool_input"]
        try:
            result=str(eval(query))
        except:
            result="Invalid calculation"
        print(f"[Calculator Tool] Result: {result}")
        return{"tool_output": result}

#not a real weather app, can implement a real one using api
class Weather:
    def __call__(self, state: ToolState):
        location=state["tool_input"]
        result=f"the weather at location {location} is sunny and 25C"
        print(f"[Weatherr Tool] Result: {result}")
        return{"tool_output": result}

#use a llm for better result formatting
class ToolResultFormatter:
    def __call__(self, state: ToolState):
        output=state["tool_output"]
        print(f"[Formatter] Tool Output: {output}")
        return{}

def create_tool_subgraph():
    sg = StateGraph(ToolState)
    sg.add_node("selector", ToolSelector(llm))
    sg.add_node("calculator", Calculator())
    sg.add_node("weather", Weather())
    sg.add_node("formatter", ToolResultFormatter())

    sg.add_edge(START, "selector")
    sg.add_conditional_edges(
        "selector",
        lambda x: x["tool_name"],
        {
            "calculator": "calculator",
            "weather": "weather",
            "none": END
        }
    )
    sg.add_edge("calculator", "formatter")
    sg.add_edge("weather", "formatter")
    sg.add_edge("formatter", END)
    return sg.compile()

if __name__ =="__main__":
    graph = create_tool_subgraph()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "tool_name": "",
            "tool_input": "",
            "tool_output": ""
        })