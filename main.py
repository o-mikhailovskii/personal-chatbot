import sys


def start_tk_interface():
    """Start the Tkinter-based chatbot interface."""
    from tkinter import Tk

    from personal_chatbot.chatbot_gui import Chatbot

    root = Tk()
    chatbot = Chatbot(root)
    root.protocol("WM_DELETE_WINDOW", chatbot.on_exit)
    root.mainloop()


def start_gradio_interface():
    """Start the Gradio-based chatbot interface."""
    from personal_chatbot.chatbot_gr import GradioChatbot

    chatbot = GradioChatbot()
    chatbot.launch()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        start_gradio_interface()
    else:
        start_tk_interface()
