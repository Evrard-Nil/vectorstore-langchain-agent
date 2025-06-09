from dotenv import load_dotenv
load_dotenv()
import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "input.json") -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_path}: {e}")
        except IOError as e:
            logger.error(f"Error reading {config_path}: {e}")
    else:
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
    return config

def initialize_llm(config: Dict[str, Any]) -> BaseLanguageModel:
    """Initialize the language model based on configuration."""
    model_name = os.getenv('DEFAULT_MODEL_NAME', config.get('llm_model', 'claude-opus-4-20250514'))
    temperature = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 2000)

    try:
        # Assuming ChatOpenAI is the primary LLM interface based on the example.
        # In a more complex scenario, this would be conditional based on model_name.
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Error initializing LLM with model '{model_name}': {e}. Attempting with default ChatOpenAI settings.")
        # Fallback to a generic ChatOpenAI if specific model fails
        return ChatOpenAI(temperature=temperature, max_tokens=max_tokens)

def initialize_tools(config: Dict[str, Any], llm: BaseLanguageModel) -> List[BaseTool]:
    """Initialize tools based on configuration."""
    tools = []
    default_tools_list = config.get("default_tools", ["search", "calculator"])
    
    for tool_name in default_tools_list:
        try:
            if tool_name == "search":
                # from langchain_community.tools import DuckDuckGoSearchRun # Already imported
                tools.append(DuckDuckGoSearchRun())
                logger.info("Added 'search' tool (DuckDuckGoSearchRun).")
            elif tool_name == "calculator":
                # from langchain.chains import LLMMathChain # Already imported
                # from langchain.tools import Tool # Already imported
                llm_math = LLMMathChain.from_llm(llm)
                tools.append(Tool(
                    name="Calculator",
                    description="useful for mathematical calculations",
                    func=llm_math.run
                ))
                logger.info("Added 'calculator' tool (LLMMathChain).")
            else:
                logger.warning(f"Unknown tool requested: {tool_name}. Skipping.")
        except ImportError as e:
            logger.error(f"Failed to import required module for tool '{tool_name}': {e}. Skipping tool.")
        except Exception as e:
            logger.error(f"Error initializing tool '{tool_name}': {e}. Skipping tool.")
    return tools

def initialize_agent(llm: BaseLanguageModel, tools: List[BaseTool], config: Dict[str, Any]):
    """Initialize the LangChain agent."""
    try:
        # Define the prompt for the agent. Using a generic ReAct pattern.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        # Create the ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Configure AgentExecutor based on config
        agent_executor_config = config.get("agent_config", {})
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=agent_executor_config.get("verbose", True),
            max_iterations=agent_executor_config.get("max_iterations", 10),
            early_stopping_method=agent_executor_config.get("early_stopping_method", "generate")
        )
        logger.info("LangChain agent initialized successfully.")
        return agent_executor
    except Exception as e:
        logger.critical(f"Failed to initialize agent: {e}")
        raise

def save_output(output_data: Dict[str, Any], output_path: str):
    """Saves the output data to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Output saved successfully to {output_path}")
    except IOError as e:
        logger.error(f"Error saving output to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a LangChain agent.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text or path to a file containing input text.")
    parser.add_argument("--output", type=str, default="output.json",
                        help="Path to the output JSON file (default: output.json).")
    parser.add_argument("--config", type=str, default="input.json",
                        help="Path to the configuration JSON file (default: input.json).")

    args = parser.parse_args()

    config = load_config(args.config)
    
    input_text = ""
    try:
        if os.path.exists(args.input) and os.path.isfile(args.input):
            with open(args.input, 'r') as f:
                input_text = f.read()
            logger.info(f"Input read from file: {args.input}")
        else:
            input_text = args.input
            logger.info("Input provided as direct text.")
    except IOError as e:
        logger.error(f"Could not read input file {args.input}: {e}")
        # Fallback to default input from config if file read fails
        input_text = config.get('default_input', 'Hello, how can you help me?') 
        logger.info(f"Using default input from config: {input_text}")
    
    if not input_text:
        input_text = config.get('default_input', 'Hello, how can you help me?')
        logger.warning(f"No input provided or read. Using default input: '{input_text}'")

    result = None
    error_message = None
    tools_initialized_names: List[str] = []

    try:
        llm = initialize_llm(config)
        tools = initialize_tools(config, llm)
        tools_initialized_names = [tool.name for tool in tools] # Capture names for metadata
        
        agent_executor = initialize_agent(llm, tools, config)
        
        logger.info(f"Executing agent with input: '{input_text}'")
        # AgentExecutor.invoke expects a dictionary with 'input' key and optionally 'chat_history'
        agent_result = agent_executor.invoke({"input": input_text, "chat_history": []})
        result = agent_result.get("output")
        logger.info("Agent execution completed.")

    except Exception as e:
        error_message = f"An unexpected error occurred during agent execution: {e}"
        logger.exception(error_message)
        result = f"Error: {error_message}"

    output_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "input_provided": input_text,
        "output": result
    }

    if config.get("include_metadata", True):
        output_data["metadata"] = {
            "llm_model_used": os.getenv('DEFAULT_MODEL_NAME', config.get('llm_model', 'claude-opus-4-20250514')),
            "temperature": config.get('temperature', 0.7),
            "max_tokens": config.get('max_tokens', 2000),
            "tools_initialized": tools_initialized_names,
            "agent_config_used": config.get("agent_config", {}),
            "error": error_message if error_message else None
        }

    save_output(output_data, args.output)

if __name__ == "__main__":
    main()