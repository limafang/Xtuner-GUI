import gradio as gr

CSS = r"""
.duplicate-button {
  margin: auto !important;
  color: white !important;
  background: black !important;
  border-radius: 100vh !important;
}

.modal-box {
  position: fixed !important;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* center horizontally */
  max-width: 1000px;
  max-height: 750px;
  overflow-y: auto;
  background-color: var(--input-background-fill);
  flex-wrap: nowrap !important;
  border: 2px solid black !important;
  z-index: 1000;
  padding: 10px;
}

.dark .modal-box {
  border: 2px solid white !important;
}
"""

chat_templates = ['internlm2-chat-7b']

with gr.Blocks(title="Xtuner Chat Board", css=CSS) as demo:
    gr.HTML(
        "<h1><center>Xtuner Chat Board</center></h1>"
    )
    with gr.Row():
        lang = gr.Dropdown(label='langguage', choices=["en", "zh"], scale=1)
        chat_template = gr.Dropdown(
            label='CHAT_TEMPLATE', choices=chat_templates, scale=3)
        predict_way = gr.Dropdown(label='predict_way',choices=['HFBot','LMDeployBot','VllmBot','OpenaiBot'])
    model_path = gr.Textbox(label='model_path', scale=3)
    chatbot = gr.Chatbot()
    history = gr.State([])

demo.queue()
demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
