from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os

from app import initialize_tools, load_config, initialize_llm

from dotenv import load_dotenv

load_dotenv()

def create_llm(model_name=os.getenv("DEFAULT_MODEL_NAME"), temperature=0, max_tokens=None):
    """
    Create and return an LLM instance.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )


# Agent initialization code will be inserted here
from langchain.agents.chat.base import ChatAgent
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool, BaseTool


config = load_config("input.json")
llm = initialize_llm(config)
# List of tools
tools: List[BaseTool] = initialize_tools(config, llm)

# Create agent template
def create_agent(llm_instance):
    chat_agent = ChatAgent.from_llm_and_tools(
        llm=llm_instance,
        tools=tools
    )

    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


def run_agent_with_input(input_text: str):
    """
    Run the agent with the given input text.

    Args:
        input_text: The input text to run the agent with

    Returns:
        The agent's response
    """
    # Create a new agent with a fresh LLM
    llm = create_llm()

    # The following code will be replaced by the generated code
    # It must use input_text and return output


    try:
        # Create agent executor
        agent_executor = create_agent(llm)

        # Run the agent
        result = agent_executor.invoke({"input": input_text})

        # Extract the output
        if isinstance(result, dict) and "output" in result:
            output = result["output"]
        else:
            output = str(result)

    except Exception as e:
        output = f"Error running chat agent: {str(e)}"

    return output

app = FastAPI()


class RunRequest(BaseModel):
    input: str
    chat_history: Optional[List[Dict[str, Any]]] = None


@app.post("/run")
async def run_agent(request: RunRequest):
    """
    Run the agent with the given input.
    """
    try:
        output = run_agent_with_input(request.input)
        return {"status": "success", "output": output}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    return {"status": "ok", "agent": "chat"}


@app.get("/tools")
async def get_tools():
    """
    Return the list of tools supported by this agent.
    """
    with open("tools.json", "r") as f:
        tools = json.load(f)
    return tools


if __name__ == "__main__":
    # Run the FastAPI app inside the docker container
    uvicorn.run(app, host="0.0.0.0", port=8000)