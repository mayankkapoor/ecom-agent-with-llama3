from main import chatbot_interface, get_chat_generator, chat_template, ChatMessage, gr

def run_app():
    messages = [ChatMessage.from_system(chat_template)]
    chat_generator = get_chat_generator()

    with gr.Blocks() as demo:
        gr.Markdown("# AI Purchase Assistant")
        gr.Markdown("Ask me about products you want to buy!")

        state = gr.State(value=messages)

        with gr.Row():
            user_input = gr.Textbox(label="Your message:")
            response_output = gr.Markdown(label="Response:")

        user_input.submit(chatbot_interface, [user_input, state], [
                          response_output, state])
        gr.Button("Send").click(chatbot_interface, [
            user_input, state], [response_output, state])

    demo.launch()

if __name__ == "__main__":
    run_app()