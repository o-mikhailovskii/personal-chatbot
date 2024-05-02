from tkinter import Tk

from personal_chatbot.chatbot_gui import Chatbot


def start_main_window():
    root = Tk()
    chatbot = Chatbot(root)
    root.protocol("WM_DELETE_WINDOW", chatbot.on_exit)
    root.mainloop()


if __name__ == "__main__":
    start_main_window()
