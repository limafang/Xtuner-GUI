from xtuner.chat import BaseChat
from xtuner.chat.templates import CHAT_TEMPLATE
import gradio as gr
import time
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

chat_templates = ['internlm2-chat-7b', 'internlm_chat']


def lang_change(lang):
    if lang == "en":
        return gr.update(label='language'), \
            gr.update(label='chat_TEMPLATE'), \
            gr.update(label='model_path'), \
            gr.update(label='predict_way'), \
            gr.update(label='Chatbot'), \
            gr.update(label='Textbox'), \
            gr.update(value='Clear'), \
            gr.update(value='init_chatbot'), \
            gr.update(label='init_name')
    elif lang == "zh":
        return gr.update(label='语言'), \
            gr.update(label='模型模板'), \
            gr.update(label='模型路径'), \
            gr.update(label='预测方式'), \
            gr.update(label='聊天机器人'), \
            gr.update(label='对话框'), \
            gr.update(value='清除记录'), \
            gr.update(value='初始化模型'), \
            gr.update(label='初始化字段')


def fn_init_chatbot(chat_TEMPLATE, predict_way, model_path):
    print(chat_TEMPLATE, predict_way, model_path)
    if predict_way == 'HFBot':
        print('---------Using HFBot!-------')
        from xtuner.chat import HFBot
        templates = CHAT_TEMPLATE[chat_TEMPLATE]
        bot = HFBot(model_path)
        global xtuner_chat_bot
        xtuner_chat_bot = BaseChat(bot, '嬛嬛', templates)
    print('init_over!')
    return gr.update(interactive=True)

def user(user_message, history):
    return "", history + [[user_message, None]]

def get_respond(chat_history):
    message = chat_history[-1][0]
    bot_message = xtuner_chat_bot.chat(message)
    chat_history[-1][1] = ""
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history



with gr.Blocks(title="Xtuner Chat Board", css=CSS) as demo:
    gr.HTML(
        "<h1><center>Xtuner Chat Board</center></h1>"
    )
    with gr.Row():
        lang = gr.Dropdown(label='language', choices=[
                           "en", "zh"], scale=1, value='en', interactive=True)
        chat_TEMPLATE = gr.Dropdown(
            label='chat_TEMPLATE', choices=chat_templates, scale=2, value='internlm2-chat-7b', interactive=True)
        init_name = gr.Textbox(label='init_name', interactive=True)
        predict_way = gr.Dropdown(label='predict_way', choices=[
                                  'HFBot', 'LMDeployBot', 'VllmBot', 'OpenaiBot'], value='HFBot', interactive=True)
        init_chatbot = gr.Button(value='init_chatbot')
    model_path = gr.Textbox(label='model_path', scale=3, interactive=True)
    chatbot = gr.Chatbot(label='Chatbot')
    history = gr.State([])
    msg = gr.Textbox(label='Textbox', interactive=False)
    clear = gr.ClearButton([msg, chatbot], value='Clear')

    lang.select(fn=lang_change, inputs=[lang],
                outputs=[lang, chat_TEMPLATE, model_path, predict_way, chatbot, msg, clear, init_chatbot, init_name])

    init_chatbot.click(fn_init_chatbot, inputs=[
                       chat_TEMPLATE, predict_way, model_path], outputs=[msg])

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(get_respond, chatbot, chatbot)

demo.queue()
demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
