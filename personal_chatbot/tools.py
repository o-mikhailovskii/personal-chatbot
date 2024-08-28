def chat_history_to_string(chat_history):
    return "\n".join(
        [
            f"{str(msg.type).upper()}: {str(msg.content).strip()}\n"
            for msg in chat_history
        ]
    )
