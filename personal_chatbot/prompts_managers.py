import json
import subprocess


class PromptDumpManager:
    def __init__(self, filename, is_dict=False):
        self.filename = filename
        self.content = self._load_lines(is_dict)
        self._content_modified = False

    def _load_lines(self, is_dict):
        try:
            with open(self.filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {} if is_dict else []

    def modify_content(self):
        self._content_modified = True

    def dump_content(self):
        if self._content_modified:
            with open(self.filename, "w") as file:
                json.dump(self.content, file)

    def dump_as_plain_text(self):
        if self._content_modified:
            with open(self.filename.replace(".json", ".txt"), "w") as file:
                for element in self.content:
                    file.write(element)
            subprocess.Popen(["open", self.filename.replace(".json", ".txt")])

    def get_prompts(self):
        return self.content


class SystemPromptSelector(PromptDumpManager):
    def __init__(self):
        super().__init__("system_prompts.json", True)


class UserPromptSelector(PromptDumpManager):
    def __init__(self):
        super().__init__("user_prompts.json")


class ChatHistoryPrompts(PromptDumpManager):
    def __init__(self, filename="chat_history.json"):
        super().__init__(filename)
