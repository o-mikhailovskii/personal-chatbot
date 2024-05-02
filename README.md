# AI Chatbot

This AI Chatbot application provides a direct way to interact with large language models (LLMs) while offering control over the system prompt and temperature settings. Unlike other clients that rely on user prompts for adjustments, this app prioritizes direct control and customization.

## Features

- **Choose your LLM provider:** Select from various providers, including Cohere, Anthropic, Google Generative AI, and more.
- **Customize the system prompt:** Tailor the initial instructions and context for the chatbot's responses.
- **Fine-tune the temperature:** Control the "creativity" and randomness of the chatbot's output.
- **Save and review chat history:** Preserve conversations and reload them later.
- **Clear chatbot memory:** Start fresh conversations with a clean slate.

## Prerequisites

- Python 3.9 or newer
- Tkinter library
- python-dotenv
- LangChain libraries (for Anthropic, Cohere, Google GenAI, Groq, Ollama, etc.)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/o-mikhailovskii/personal-chatbot.git
   ```

2. Navigate to the project directory:
   ```bash
   cd personal-chatbot
   ```

3. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   poetry install --no-root
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
   Alternatively, consider creating a macOS Automation for easier access.

2. Choose your desired language model provider and click "Start."

3. Interact with the chatbot:
   - Edit the "System Prompt" to provide initial instructions.
   - Adjust the "Temperature" value to influence response creativity.
   - Type your message in the "User Prompt" section and click "Send."
   - View the conversation history in the "AI Response" section.
   - Click "Clear Memory" to start a new conversation.
   - Save the conversation using the "Save Chat History" button.

## Contributing

Contributions and feedback are welcome! Please open an issue or submit a pull request if you encounter any problems or have suggestions for improvement.

## Future Improvements

- Better customization of saved `SYSTEM` and `USER` prompts.
- Add proper tests.

## Important Note

Please review and respect the terms of service of your chosen LLM provider. Avoid submitting any sensitive or personal information during your interactions, if unsure of how the provider handles it.

## Acknowledgments

A significant portion of the prompts in this repository were sourced from Reddit, X/Twitter, and official guides, including the Claude prompts library.
