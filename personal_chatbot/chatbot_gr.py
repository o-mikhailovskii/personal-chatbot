import gradio as gr

from .llm_chain_manager import LLM_PROVIDERS, LLMChainManager
from .prompts_managers import (
    ChatHistoryPrompts,
    ConfigManager,
    SystemPromptSelector,
    UserPromptSelector,
)


class GradioChatbot:
    def __init__(self):
        self.user_prompts_manager = UserPromptSelector()
        self.custom_system_prompts_manager = SystemPromptSelector()
        self.chat_history_manager = ChatHistoryPrompts()
        self.config_manager = ConfigManager()

        self.system_prompt_options = list(
            self.custom_system_prompts_manager.get_prompts().keys()
        )
        self.user_prompt_options = list(self.user_prompts_manager.get_prompts().keys())

        self.chat_history = []
        self.llm_chain_manager = None

        self.default_system_prompt_key = "default"
        (
            self.default_temperature,
            self.default_system_prompt,
        ) = self.custom_system_prompts_manager.get_prompts()[
            self.default_system_prompt_key
        ]
        self.engine = None
        self.temperature = 0
        self.system_prompt = None

    def load_config(self):
        try:
            config = self.config_manager.get_prompts()
            self.engine = config.get("engine", None)
            self.temperature = config.get("temperature", self.default_temperature)
            self.system_prompt = config.get("system_prompt", self.default_system_prompt)
            self.init_llm_chain_manager()
            return (
                "Config loaded successfully.",
                self.engine,
                self.temperature,
                self.system_prompt,
            )
        except Exception as e:
            return f"Error: Failed to load config: {str(e)}"

    def init_llm_chain_manager(self):
        self.llm_chain_manager = LLMChainManager(
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )
        self.llm_chain_manager.init_llm(self.engine)
        self.llm_chain_manager.init_prompt()
        self.llm_chain_manager.init_memory()
        self.llm_chain_manager.init_llm_chain()

    def choose_engine(self, engine):
        self.engine = engine
        self.init_llm_chain_manager()
        return f"Engine set to: {engine}"

    def change_system_prompt(self, new_prompt):
        if new_prompt:
            self.system_prompt = new_prompt
            self.init_llm_chain_manager()
            self.clear_memory()
            return "System prompt updated successfully."
        else:
            return "Error: System prompt cannot be empty."

    def set_system_prompt(self, selected_prompt):
        (
            self.temperature,
            self.system_prompt,
        ) = self.custom_system_prompts_manager.get_prompts()[selected_prompt]
        self.init_llm_chain_manager()
        self.clear_memory()
        return (
            f"System prompt set to: {selected_prompt}",
            str(self.temperature),
            self.system_prompt,
        )

    def set_user_prompt(self, selected_prompt):
        prompt_text = self.user_prompts_manager.get_prompts()[selected_prompt]
        return prompt_text

    def change_temperature(self, new_temperature):
        try:
            new_temperature = float(new_temperature)
            if 0 <= new_temperature <= 1:
                self.temperature = new_temperature
                self.llm_chain_manager.init_llm(self.engine)
                self.llm_chain_manager.init_llm_chain()
                return "Temperature updated successfully."
            else:
                return "Error: Temperature must be between 0 and 1."
        except ValueError:
            return "Error: Invalid input. Please enter a valid number."

    def send_message(self, user_input):
        if user_input:
            try:
                response = self.llm_chain_manager.llm_chain.predict(
                    human_input=user_input
                )
                self.chat_history.append(f"USER: {user_input}")
                self.chat_history.append(f"AI: {response}")
                return "\n".join(self.chat_history)
            except Exception as e:
                return f"Error: An error occurred: {str(e)}"
        else:
            return "Error: User input cannot be empty."

    def clear_memory(self):
        self.llm_chain_manager.memory.clear()
        self.chat_history = []
        return "Memory cleared."

    def save_chat_history(self):
        try:
            self.chat_history_manager.content = self.chat_history
            self.chat_history_manager.modify_content()
            self.chat_history_manager.dump_as_plain_text()
            return "Chat history saved successfully."
        except Exception as e:
            return f"Error: Failed to save chat history: {str(e)}"

    def save_config(self):
        try:
            config = {
                "engine": self.engine,
                "temperature": self.temperature,
                "system_prompt": self.system_prompt,
            }
            self.config_manager.content = config
            self.config_manager.modify_content()
            self.config_manager.dump_content()
            return "Config saved successfully."
        except Exception as e:
            return f"Error: Failed to save config: {str(e)}"

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("# AI Chatbot")

            with gr.Group():
                with gr.Row():
                    load_config_button = gr.Button("Load Config")
                    save_config_button = gr.Button("Save Config")

                with gr.Row():
                    engine_dropdown = gr.Dropdown(
                        choices=sorted(LLM_PROVIDERS.keys()), label="Choose Engine"
                    )
                    temperature_input = gr.Textbox(
                        value=str(self.temperature), label="Temperature"
                    )
                    system_prompt_dropdown = gr.Dropdown(
                        choices=self.system_prompt_options,
                        label="System Prompt Library",
                    )
                with gr.Row():
                    engine_button = gr.Button("Set Engine")
                    temperature_button = gr.Button("Change Temperature")
                    set_system_prompt_button = gr.Button("Set System Prompt")

            # with gr.Group():
            with gr.Column():
                system_prompt_input = gr.Textbox(
                    value=self.system_prompt, label="System Prompt", lines=5
                )
                system_prompt_button = gr.Button("Change System Prompt")

            with gr.Group():
                with gr.Row():
                    user_prompt_dropdown = gr.Dropdown(
                        choices=self.user_prompt_options, label="User Prompt Library"
                    )
                    set_user_prompt_button = gr.Button("Set User Prompt")

            with gr.Group():
                user_input = gr.Textbox(label="User Input", lines=5)
                send_button = gr.Button("Send")
                output = gr.Textbox(label="Chat History", lines=10)

            with gr.Group():
                clear_button = gr.Button("Clear Memory")
                save_button = gr.Button("Save Chat History")

            # Connect components

            status_output = gr.Textbox(label="Status")
            load_config_button.click(
                self.load_config,
                outputs=[
                    status_output,
                    engine_dropdown,
                    temperature_input,
                    system_prompt_input,
                ],
            )
            save_config_button.click(self.save_config, outputs=[status_output])
            engine_button.click(
                self.choose_engine,
                inputs=[engine_dropdown],
                outputs=[status_output],
            )
            system_prompt_button.click(
                self.change_system_prompt,
                inputs=[system_prompt_input],
                outputs=[status_output],
            )
            set_system_prompt_button.click(
                self.set_system_prompt,
                inputs=[system_prompt_dropdown],
                outputs=[status_output, temperature_input, system_prompt_input],
            )
            set_user_prompt_button.click(
                self.set_user_prompt,
                inputs=[user_prompt_dropdown],
                outputs=[user_input],
            )
            temperature_button.click(
                self.change_temperature,
                inputs=[temperature_input],
                outputs=[status_output],
            )
            send_button.click(self.send_message, inputs=[user_input], outputs=[output])
            clear_button.click(self.clear_memory, outputs=[status_output])
            save_button.click(self.save_chat_history, outputs=[status_output])

        demo.launch()
