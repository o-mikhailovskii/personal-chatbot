from tkinter import (
    END,
    Button,
    Entry,
    Label,
    LabelFrame,
    OptionMenu,
    StringVar,
    Text,
    Toplevel,
    messagebox,
)
from tkinter.scrolledtext import ScrolledText

from .llm_chain_manager import LLM_PROVIDERS, LLMChainManager
from .prompts_managers import (
    ChatHistoryPrompts,
    SystemPromptSelector,
    UserPromptSelector,
)

PAD = 2


# GUI Class
class Chatbot:
    def __init__(self, root):
        # Start the GUI
        self.root = root
        self.selected_option = StringVar()

        # Initialize prompts managers
        self.user_prompts_manager = UserPromptSelector()
        self.custom_system_prompts_manager = SystemPromptSelector()
        self.chat_history_manager = ChatHistoryPrompts()

        self.system_prompt_options = (
            self.custom_system_prompts_manager.get_prompts().keys()
        )
        self.user_prompt_options = self.user_prompts_manager.get_prompts().keys()

        # Choose engine
        self._choose_engine()

    def _run_engine(self):
        self.root.title(f"{self.engine} AI Chatbot")
        self.chat_history = []
        self.default_system_prompt_key = "default"
        (
            self.temperature,
            self.system_prompt,
        ) = self.custom_system_prompts_manager.get_prompts()[
            self.default_system_prompt_key
        ]

        # Initialize GUI elements
        self.create_gui_elements()

        # Bind the on_exit method to the window's destroy event
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Initialize LLMChain manager
        self.llm_chain_manager = None
        self.init_llm_chain_manager()

    def _choose_engine(self):
        self.root.title("Engine Selector")
        options = sorted(LLM_PROVIDERS.keys())
        self.selected_option.set(options[0])

        # Create a drop-down menu for the selector
        option_menu = OptionMenu(self.root, self.selected_option, *options)
        option_menu.pack()

        def clean_window(engine):
            self.engine = engine
            for widget in self.root.winfo_children():
                widget.destroy()
            self._run_engine()

        # Create a button to start the main window
        start_button = Button(
            self.root,
            text="Start",
            command=lambda: clean_window(self.selected_option.get()),
        )
        start_button.pack()

    def create_text_element(
        self, cv, height, width, default_text="", scrolled=False, side=None
    ):
        """
        Create a Text widget and pack it with the specified height, and width.

        Args:
            cv (tkinter widget): The container widget in which the Text widget will be packed. # noqa: E501
            height (int): The height of the Text widget in lines.
            width (int): The width of the Text widget in characters.
            default_text (str, optional): The default text to display in the Text widget. Defaults to an empty string. # noqa: E501
            scrolled (bool, optional): Whether to use a ScrolledText widget. Default is False. # noqa: E501
            side (str, optional): The side to pack the Text widget. Default is None.

        Returns:
            Text or ScrolledText: The created Text or ScrolledText widget.
        """
        if scrolled:
            text_element = ScrolledText(cv, height=height, width=width)
        else:
            text_element = Text(cv, height=height, width=width)

        if side:
            text_element.pack(side=side, pady=PAD)
        else:
            text_element.pack(pady=PAD)
        text_element.insert(END, default_text)

        return text_element

    def create_button_element(self, cv, label, command):
        """
        Create a Button widget and pack it with the specified label and command.

        Args:
            cv (tkinter widget): The container widget in which the Button widget will be packed. # noqa: E501
            label (str): The label to display on the Button widget.
            command: The command to execute when the Button widget is clicked.

        Returns:
            Button: The created Button widget.
        """
        button_element = Button(cv, text=label, command=command)
        button_element.pack(side="left", pady=PAD)

        return button_element

    def add_system_prompt(self):
        """Opens a dialog to add a new system prompt."""

        def save_new_prompt():
            prompt_name = entry_name.get()
            prompt_text = entry_text.get("1.0", END)
            temperature = entry_temp.get()

            if not prompt_name or not prompt_text:
                messagebox.showerror("Error", "Please enter a name and prompt text.")
                return

            try:
                temperature = float(temperature)
                if not (0 <= temperature <= 1):
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Error", "Temperature must be a number between 0 and 1."
                )
                return

            self.custom_system_prompts_manager.content[prompt_name] = (
                temperature,
                prompt_text,
            )

            self.custom_system_prompts_manager.modify_content()

            # Update system prompt options
            self.system_prompt_options = (
                self.custom_system_prompts_manager.get_prompts().keys()
            )
            self.selected_system_prompt.set(prompt_name)  # Select the new prompt

            # Update dropdown menu
            menu = self.system_prompt_dropdown["menu"]
            menu.delete(0, END)
            for option in self.system_prompt_options:
                menu.add_command(
                    label=option,
                    command=lambda v=option: self.selected_system_prompt.set(v),
                )

            messagebox.showinfo("Success", "System prompt added successfully.")
            add_prompt_window.destroy()

        add_prompt_window = Toplevel(self.root)
        add_prompt_window.title("Add System Prompt")

        Label(add_prompt_window, text="Name:").pack()
        entry_name = Entry(add_prompt_window)
        entry_name.pack()

        Label(add_prompt_window, text="Prompt Text:").pack()
        entry_text = Text(add_prompt_window, height=10)
        entry_text.insert(END, self.system_prompt)
        entry_text.pack()

        Label(add_prompt_window, text="Temperature (0-1):").pack()
        entry_temp = Entry(add_prompt_window)
        entry_temp.pack()

        Button(add_prompt_window, text="Save", command=save_new_prompt).pack()

    def add_user_prompt(self):
        """Opens a dialog to add a new user prompt."""

        def save_new_prompt():
            prompt_name = entry_name.get()
            prompt_text = entry_text.get("1.0", END)

            if not prompt_name or not prompt_text:
                messagebox.showerror("Error", "Please enter a name and prompt text.")
                return

            self.user_prompts_manager.content[prompt_name] = prompt_text
            self.user_prompts_manager.modify_content()

            # Update user prompt options
            self.user_prompt_options = self.user_prompts_manager.get_prompts().keys()
            self.selected_user_prompt.set(prompt_name)  # Select the new prompt

            # Update dropdown menu
            menu = self.user_prompt_dropdown["menu"]
            menu.delete(0, END)
            for option in self.user_prompt_options:
                menu.add_command(
                    label=option,
                    command=lambda v=option: self.selected_user_prompt.set(v),
                )

            messagebox.showinfo("Success", "User prompt added successfully.")
            add_prompt_window.destroy()

        add_prompt_window = Toplevel(self.root)
        add_prompt_window.title("Add User Prompt")

        Label(add_prompt_window, text="Name:").pack()
        entry_name = Entry(add_prompt_window)
        entry_name.pack()

        Label(add_prompt_window, text="Prompt Text:").pack()
        entry_text = Text(add_prompt_window, height=10)
        entry_text.insert(END, self.input_box.get("1.0", "end-1c"))
        entry_text.pack()

        Button(add_prompt_window, text="Save", command=save_new_prompt).pack()

    def create_system_prompt_section(self):
        # System Prompt Selection
        system_prompt_frame = LabelFrame(self.root, text="System Prompt")
        system_prompt_frame.pack(pady=PAD)

        self.system_prompt_box = self.create_text_element(
            system_prompt_frame, 10, 120, self.system_prompt
        )
        self.change_system_prompt_button = self.create_button_element(
            system_prompt_frame, "Change System Prompt", self.change_system_prompt
        )

        self.selected_system_prompt = StringVar()
        self.selected_system_prompt.set(self.default_system_prompt_key)

        # System Prompt Library Selection
        spl_frame = LabelFrame(system_prompt_frame, text="System Prompt Library")
        spl_frame.pack(side="left", pady=PAD)

        self.system_prompt_dropdown = OptionMenu(
            spl_frame, self.selected_system_prompt, *self.system_prompt_options
        )
        self.system_prompt_dropdown.pack(side="left", pady=PAD)

        change_system_prompt_button = Button(
            spl_frame, text="Set", command=self.set_system_prompt
        )
        change_system_prompt_button.pack(side="left", pady=PAD)

        edit_system_prompt_button = Button(
            spl_frame,
            text="Edit",
            command=self.edit_system_prompt,
        )
        edit_system_prompt_button.pack(side="left", pady=PAD)

        # Temperature Selection
        temperature_frame = LabelFrame(system_prompt_frame, text="Temperature")
        temperature_frame.pack(side="left", pady=PAD)

        self.temperature_box = self.create_text_element(
            temperature_frame, 1, 4, self.temperature, side="left"
        )
        self.change_temperature_button = self.create_button_element(
            temperature_frame, "Change Temperature", self.change_temperature
        )

        add_system_prompt_button = Button(
            spl_frame, text="Add", command=self.add_system_prompt
        )
        add_system_prompt_button.pack(side="left", pady=PAD)

    def create_user_input_section(self):
        # User Input
        user_frame = LabelFrame(self.root, text="User Prompt")
        user_frame.pack(pady=PAD)

        self.input_box = self.create_text_element(user_frame, 10, 120, scrolled=True)
        self.send_message_button = self.create_button_element(
            user_frame, "Send", self.send_message
        )

        # User Prompt Library Selection
        upl_frame = LabelFrame(user_frame, text="User Prompt Library")
        upl_frame.pack(side="left", pady=PAD)

        self.selected_user_prompt = StringVar()
        self.selected_user_prompt.set(
            sorted(self.user_prompt_options)[0]
        )  # Default selection

        self.user_prompt_dropdown = OptionMenu(
            upl_frame, self.selected_user_prompt, *self.user_prompt_options
        )
        self.user_prompt_dropdown.pack(side="left", pady=PAD)

        set_user_prompt_button = Button(
            upl_frame, text="Set", command=self.set_user_prompt
        )
        set_user_prompt_button.pack(side="left", pady=PAD)

        edit_user_prompt_button = Button(
            upl_frame, text="Edit", command=self.edit_user_prompt
        )
        edit_user_prompt_button.pack(side="left", pady=PAD)

        add_user_prompt_button = Button(
            upl_frame, text="Add", command=self.add_user_prompt
        )
        add_user_prompt_button.pack(side="left", pady=PAD)

    def create_ai_response_section(self):
        # AI Response
        ai_response_frame = LabelFrame(self.root, text="AI Response")
        ai_response_frame.pack(pady=PAD)

        self.output_box = self.create_text_element(
            ai_response_frame, 20, 120, scrolled=True
        )
        self.clear_memory_button = self.create_button_element(
            ai_response_frame, "Clear Memory", self.clear_memory
        )
        self.save_history_button = self.create_button_element(
            ai_response_frame, "Save Chat History", self.save_chat_history
        )

    def create_gui_elements(self):
        """
        Create and pack all the GUI elements.
        """
        self.create_system_prompt_section()
        self.create_user_input_section()
        self.create_ai_response_section()

    def init_llm_chain_manager(self):
        """
        Initialize the LLMChain manager with the current system prompt and temperature.
        """
        self.llm_chain_manager = LLMChainManager(
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )
        self.llm_chain_manager.init_llm(self.engine)
        self.llm_chain_manager.init_prompt()
        self.llm_chain_manager.init_memory()
        self.llm_chain_manager.init_llm_chain()

    def change_system_prompt(self):
        """
        Change the system prompt and update the LLMChain manager.
        """
        new_prompt = self.system_prompt_box.get("1.0", "end-1c")
        if new_prompt:
            self.system_prompt = new_prompt
            self.init_llm_chain_manager()
            self.clear_memory()
            messagebox.showinfo("Success", "System prompt updated successfully.")
        else:
            messagebox.showerror("Error", "System prompt cannot be empty.")

    def set_system_prompt(self):
        """
        Sets the system prompt based on the selected option.
        """
        (
            self.temperature,
            self.system_prompt,
        ) = self.custom_system_prompts_manager.get_prompts()[
            self.selected_system_prompt.get()
        ]
        self.temperature_box.delete("1.0", END)
        self.temperature_box.insert(END, self.temperature)
        self.system_prompt_box.delete("1.0", END)
        self.system_prompt_box.insert(END, self.system_prompt)
        self.init_llm_chain_manager()
        self.clear_memory()
        messagebox.showinfo("Success", "System prompt updated successfully.")

    def set_user_prompt(self):
        """
        Sets the user prompt in the input box based on the selected option.
        """
        selected_prompt = self.selected_user_prompt.get()
        prompt_text = self.user_prompts_manager.get_prompts()[selected_prompt]
        self.input_box.delete("1.0", END)
        self.input_box.insert(END, prompt_text)

    def edit_system_prompt(self):
        """Opens a window to edit existing system prompts."""
        edit_window = Toplevel(self.root)
        edit_window.title("Edit System Prompt")

        prompts = self.custom_system_prompts_manager.get_prompts()

        def save_edits():
            for name, (temp_entry, prompt_entry) in entries.items():
                new_temp = temp_entry.get()
                new_prompt = prompt_entry.get("1.0", END)
                try:
                    new_temp = float(new_temp)
                    if not (0 <= new_temp <= 1):
                        raise ValueError
                except ValueError:
                    messagebox.showerror(
                        "Error",
                        f"Invalid temperature for '{name}': Must be between 0 and 1.",
                    )
                    return

                prompts[name] = (new_temp, new_prompt)
            self.custom_system_prompts_manager.modify_content(prompts)
            messagebox.showinfo("Success", "System prompts updated successfully.")
            edit_window.destroy()

        entries = {}
        for name, (temperature, prompt) in prompts.items():
            frame = LabelFrame(edit_window, text=name)
            frame.pack(pady=PAD)

            Label(frame, text="Temperature (0-1):").pack()
            temp_entry = Entry(frame)
            temp_entry.insert(0, temperature)
            temp_entry.pack()

            Label(frame, text="Prompt Text:").pack()
            prompt_entry = Text(frame, height=5)
            prompt_entry.insert(END, prompt)
            prompt_entry.pack()

            entries[name] = (temp_entry, prompt_entry)

        Button(edit_window, text="Save Changes", command=save_edits).pack(pady=PAD)

    def edit_user_prompt(self):
        """Opens a window to edit existing user prompts."""
        edit_window = Toplevel(self.root)
        edit_window.title("Edit User Prompt")

        prompts = self.user_prompts_manager.get_prompts()

        def save_edits():
            for name, prompt_entry in entries.items():
                new_prompt = prompt_entry.get("1.0", END)
                prompts[name] = new_prompt
            self.user_prompts_manager.modify_content(prompts)
            messagebox.showinfo("Success", "User prompts updated successfully.")
            edit_window.destroy()

        entries = {}
        for name, prompt in prompts.items():
            frame = LabelFrame(edit_window, text=name)
            frame.pack(pady=PAD)

            Label(frame, text="Prompt Text:").pack()
            prompt_entry = Text(frame, height=5)
            prompt_entry.insert(END, prompt)
            prompt_entry.pack()

            entries[name] = prompt_entry

        Button(edit_window, text="Save Changes", command=save_edits).pack(pady=PAD)

    def change_temperature(self):
        """
        Change the temperature and update the LLMChain manager.
        """
        try:
            new_temperature = float(self.temperature_box.get("1.0", "end-1c"))
            if 0 <= new_temperature <= 1:
                self.temperature = new_temperature
                self.llm_chain_manager.init_llm(self.engine)
                self.llm_chain_manager.init_llm_chain()
                messagebox.showinfo("Success", "Temperature updated successfully.")
            else:
                messagebox.showerror("Error", "Temperature must be between 0 and 1.")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a valid number.")

    def send_message(self):
        """
        Send user input to the LLMChain for processing and display the response.
        """
        user_input = self.input_box.get("1.0", "end-1c")
        if user_input:
            try:
                # Send user input to LangChain for processing
                response = self.llm_chain_manager.llm_chain.predict(
                    human_input=user_input
                )

                # Add user input and response to chat history
                self.chat_history.append(f"USER: {user_input}\n")
                self.chat_history.append(f"  AI: {response}\n")

                # Display the response in the output box
                self.output_box.delete("1.0", END)
                for message in self.chat_history[-2:]:
                    self.output_box.insert(END, message + "\n")
                self.output_box.see(END)
                self.input_box.delete("1.0", END)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showerror("Error", "User input cannot be empty.")

    def clear_memory(self):
        """
        Clear the LLM chain memory and chat history.
        """
        self.llm_chain_manager.memory.clear()
        self.chat_history = []
        self.output_box.delete("1.0", END)

    def save_chat_history(self):
        """
        Saves the current chat history to a JSON file.
        """
        try:
            self.chat_history_manager.content = self.chat_history
            self.chat_history_manager.modify_content()
            self.chat_history_manager.dump_as_plain_text()
            messagebox.showinfo("Success", "Chat history saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save chat history: {str(e)}")

    def on_exit(self):
        """
        Cleanup objects to be executed when the GUI is closed.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.user_prompts_manager.dump_content()
            self.custom_system_prompts_manager.dump_content()
            self.root.destroy()
