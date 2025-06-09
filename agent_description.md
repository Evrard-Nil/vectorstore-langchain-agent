### Vectorstore LangChain Chat Agent

A LangChain-based chat agent designed to process natural language inputs, utilize a set of predefined tools, and generate responses. It leverages OpenAI's language models for conversational capabilities and tool execution, making it suitable for interactive applications that require dynamic tool use.

#### Main Functions

*   **Conversational AI:** Engages in chat-based interactions, processing user queries and generating relevant responses.
*   **Tool Utilization:** Integrates and executes various tools based on the conversation context to perform specific actions or retrieve information. The available tools are initialized from a configuration file (`input.json`).
*   **API Endpoints:** Provides HTTP endpoints for running the agent and retrieving its supported tools.

#### Inputs

*   **Medium:** HTTP POST request
*   **Endpoint:** `/run`
*   **Format:** JSON object with the following fields:
    *   `input` (string): The user's query or message for the agent.
    *   `chat_history` (optional, list of objects): Previous conversational turns. *Note: While accepted by the API, this agent currently only processes the `input` field for its core logic.*

#### Outputs

*   **Medium:** HTTP JSON response
*   **Endpoint:** `/run`
*   **Format:** JSON object with the following fields:
    *   `status` (string): Indicates "success" or "error".
    *   `output` (string): The agent's generated response or an error message if an issue occurred.
*   **Endpoint:** `/tools`
*   **Format:** JSON array containing definitions of the tools supported by the agent.

#### Environment Variables

*   **`OPENAI_API_KEY`**: Required for authenticating with the OpenAI API.
*   **`DEFAULT_MODEL_NAME`**: Specifies the default OpenAI model to be used (e.g., `gpt-4o`).
*   **`OPENAI_BASE_URL`**: (Optional) Specifies a custom base URL for the OpenAI API, useful for local or proxy setups.
*   **`OPENAI_ORG_ID`**: (Optional) Specifies the OpenAI organization ID.