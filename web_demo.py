from cProfile import label
from multiprocessing.sharedctypes import Value
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
def lang_change(lang):
    if lang == "en":
        return  gr.update(label='language'), \
                gr.update(label='chat_TEMPLATE'), \
                gr.update(label='model_path'), \
                gr.update(label='predict_way'), \
                gr.update(label='Chatbot'), \
                gr.update(label='Textbox'), \
                gr.update(value='Clear')
    elif lang == "zh":
        return  gr.update(label='语言'), \
                gr.update(label='模型模板'), \
                gr.update(label='模型路径'), \
                gr.update(label='预测方式'), \
                gr.update(label='聊天机器人'), \
                gr.update(label='对话框'), \
                gr.update(value='清除记录')

with gr.Blocks(title="Xtuner Chat Board", css=CSS) as demo:
    gr.HTML(
        "<h1><center>Xtuner Chat Board</center></h1>"
    )
    with gr.Row():
        lang = gr.Dropdown(label='language', choices=["en", "zh"], scale=1)
        chat_TEMPLATE = gr.Dropdown(
            label='chat_TEMPLATE', choices=chat_templates, scale=3)
        predict_way = gr.Dropdown(label='predict_way',choices=['HFBot','LMDeployBot','VllmBot','OpenaiBot'])
    model_path = gr.Textbox(label='model_path', scale=3)
    chatbot = gr.Chatbot(label='Chatbot')
    history = gr.State([])
    msg = gr.Textbox(label='Textbox')
    clear = gr.ClearButton([msg, chatbot], value='Clear')
    lang.select(fn=lang_change, inputs=[lang], outputs=[lang, chat_TEMPLATE, model_path, predict_way, chatbot, msg, clear])


demo.queue()
demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
