from dotenv import load_dotenv
load_dotenv()
import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Union

# LangChain specific imports
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_cli_path: str = None) -> Dict[str, Any]:
    """
    Loads configuration from input.json and optionally overrides with a CLI specified config file.
    """
    config = {}
    default_config_path = "input.json"

    # Load from default input.json if it exists
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, 'r') as f:
                config.update(json.load(f))
            logger.info(f"Loaded default configuration from {default_config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {default_config_path}: {e}")
        except IOError as e:
            logger.error(f"Error reading {default_config_path}: {e}")

    # Override with CLI specified config if provided
    if config_cli_path and os.path.exists(config_cli_path):
        try:
            with open(config_cli_path, 'r') as f:
                cli_config = json.load(f)
                config.update(cli_config) # CLI config overrides default
            logger.info(f"Loaded and applied CLI configuration from {config_cli_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_cli_path}: {e}")
        except IOError as e:
            logger.error(f"Error reading {config_cli_path}: {e}")
    elif config_cli_path:
        logger.warning(f"CLI config file not found: {config_cli_path}")

    # Set default values if not present
    config.setdefault("llm_model", "claude-opus-4-20250514")
    config.setdefault("temperature", 0.7)
    config.setdefault("max_tokens", 2000)
    config.setdefault("default_tools", ["search", "calculator"])
    config.setdefault("default_input", "Hello, how can you help me?")
    config.setdefault("agent_config", {"verbose": True, "max_iterations": 10, "early_stopping_method": "generate"})
    config.setdefault("output_format", "json")
    config.setdefault("include_metadata", True)
    config.setdefault("error_handling", {"log_level": "INFO", "save_errors_to_output": True})
    config.setdefault("tool_config", {"handle_import_errors": True, "warn_on_missing_tools": True})

    # Set logging level based on config
    log_level_str = config.get("error_handling", {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    return config

def initialize_llm(config: Dict[str, Any]) -> BaseLanguageModel:
    """Initialize the language model based on configuration."""
    model_name = os.getenv('DEFAULT_MODEL_NAME', config.get('llm_model', 'claude-opus-4-20250514'))
    temperature = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 2000)

    try:
        from langchain_openai import ChatOpenAI
        logger.info(f"Initializing ChatOpenAI with model: {model_name}, temp: {temperature}, max_tokens: {max_tokens}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except ImportError:
        logger.error("langchain_openai not installed. Please install it with 'pip install langchain-openai'.")
        raise
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        # Fallback or re-raise based on desired behavior. Re-raising for now.
        raise

def initialize_tools(config: Dict[str, Any], llm: BaseLanguageModel) -> List[BaseTool]:
    """Initialize tools based on configuration."""
    tools = []
    tool_config = config.get("tool_config", {})

    for tool_name in config.get("default_tools", []):
        try:
            if tool_name == "search":
                from langchain_community.tools import DuckDuckGoSearchRun
                tools.append(DuckDuckGoSearchRun())
                logger.info("Initialized DuckDuckGoSearchRun tool.")
            elif tool_name == "calculator":
                from langchain.chains import LLMMathChain
                from langchain.tools import Tool
                llm_math = LLMMathChain.from_llm(llm)
                tools.append(Tool(
                    name="Calculator",
                    description="useful for mathematical calculations",
                    func=llm_math.run
                ))
                logger.info("Initialized Calculator tool.")
            else:
                if tool_config.get("warn_on_missing_tools", True):
                    logger.warning(f"Unknown tool requested: {tool_name}. Skipping.")
        except ImportError as e:
            if tool_config.get("handle_import_errors", True):
                logger.error(f"Failed to import necessary module for tool '{tool_name}': {e}. Skipping this tool.")
            else:
                raise # Re-raise if not configured to handle import errors
        except Exception as e:
            logger.error(f"Error initializing tool '{tool_name}': {e}. Skipping this tool.")
    return tools

def initialize_agent(llm: BaseLanguageModel, tools: List[BaseTool], config: Dict[str, Any]) -> AgentExecutor:
    """Initialize the LangChain agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor_config = config.get("agent_config", {})
    
    logger.info(f"Initializing AgentExecutor with config: {agent_executor_config}")
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=agent_executor_config.get("verbose", True),
        handle_parsing_errors=True,
        max_iterations=agent_executor_config.get("max_iterations", 10),
        early_stopping_method=agent_executor_config.get("early_stopping_method", "generate")
    )
    return agent_executor

def save_output(output_data: Dict[str, Any], output_path: str):
    """Saves the output data to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Output successfully saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving output to {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving output: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a LangChain agent with configurable tools and LLM.")
    parser.add_argument("--input", type=str, help="Input text or path to a text file.")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save the output JSON file.")
    parser.add_argument("--config", type=str, help="Path to a custom JSON configuration file to override defaults.")
    args = parser.parse_args()

    output_data = {
        "result": None,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_provided": args.input,
            "output_file": args.output,
            "status": "failed"
        }
    }

    try:
        config = load_config(args.config)
        output_data["metadata"]["config_used"] = config

        # Determine input
        input_text = args.input
        if input_text is None:
            input_text = config.get("default_input", "Hello, how can you help me?")
            logger.info(f"No input provided via CLI. Using default input: '{input_text}'")
        elif os.path.exists(input_text) and os.path.isfile(input_text):
            try:
                with open(input_text, 'r') as f:
                    input_text = f.read()
                logger.info(f"Read input from file: {args.input}")
            except IOError as e:
                logger.error(f"Could not read input file {args.input}: {e}")
                output_data["metadata"]["error"] = f"Failed to read input file: {e}"
                save_output(output_data, args.output)
                return
        else:
            logger.info(f"Using direct text input: '{input_text}'")

        output_data["metadata"]["processed_input"] = input_text

        llm = initialize_llm(config)
        tools = initialize_tools(config, llm)
        agent_executor = initialize_agent(llm, tools, config)

        logger.info("Executing agent...")
        agent_result = agent_executor.invoke({"input": input_text})
        
        output_data["result"] = agent_result
        output_data["metadata"]["status"] = "success"
        logger.info("Agent execution completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during agent execution:")
        output_data["metadata"]["error"] = str(e)
        if config.get("error_handling", {}).get("save_errors_to_output", True):
            output_data["result"] = {"error": str(e), "traceback": str(e.__traceback__)} # Simplified traceback
        output_data["metadata"]["status"] = "failed"
    finally:
        save_output(output_data, args.output)

if __name__ == "__main__":
    main()