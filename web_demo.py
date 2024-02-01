from xtuner.chat import BaseChat
from xtuner.chat import CHAT_TEMPLATE
from xtuner.chat import GenerationConfig
import gradio as gr
import pandas as pd
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
df = pd.read_excel('/root/results.xlsx')

text_data = """
请给我介绍五个上海景点
请给我介绍五个北京景点
请给我介绍五个海南景点
"""

chat_templates = ['internlm_chat', 'internlm2_chat', 'zephyr', 'moss_sft', 'llama2_chat', 'code_llama_chat', 'chatglm2', 'chatglm3', 'qwen_chat',
                  'baichuan_chat', 'baichuan2_chat', 'wizardlm', 'wizardcoder', 'vicuna', 'deepseek_coder', 'deepseekcoder', 'deepseek_moe', 'mistral', 'mixtral']


def lang_change(lang):
    if lang == "en":
        return gr.update(label='language'), \
            gr.update(label='chat_TEMPLATE'), \
            gr.update(label='model_path'), \
            gr.update(label='inference_engine'), \
            gr.update(label='Chatbot'), \
            gr.update(label='Textbox'), \
            gr.update(value='Clear'), \
            gr.update(value='init_chatbot'), \
            gr.update(label='bot_name')
    elif lang == "zh":
        return gr.update(label='语言'), \
            gr.update(label='对话模板'), \
            gr.update(label='模型路径'), \
            gr.update(label='推理引擎'), \
            gr.update(label='聊天机器人'), \
            gr.update(label='对话框'), \
            gr.update(value='清除记录'), \
            gr.update(value='初始化模型'), \
            gr.update(label='模型名称')


def fn_init_chatbot(chat_TEMPLATE, inference_engine, model_path):
    print(chat_TEMPLATE, inference_engine, model_path)
    global xtuner_chat_bot
    if inference_engine == 'Huggingface':
        print('---------Using HFBot!-------')
        from xtuner.chat import HFBot
        templates = CHAT_TEMPLATE[chat_TEMPLATE]
        bot = HFBot(model_path)
        xtuner_chat_bot = BaseChat(bot, '嬛嬛', templates)
    if inference_engine == 'LMDeploy':
        print('---------Using LMDeployBot!-------')
        from xtuner.chat import LMDeployBot
        templates = CHAT_TEMPLATE[chat_TEMPLATE]
        bot = LMDeployBot(model_path)
        xtuner_chat_bot = BaseChat(bot, '嬛嬛', templates)
    if inference_engine == 'Vllm':
        print('---------Using VllmBot!-------')
        from xtuner.chat import VllmBot
        templates = CHAT_TEMPLATE[chat_TEMPLATE]
        bot = VllmBot(model_path)
        xtuner_chat_bot = BaseChat(bot, '嬛嬛', templates)
    if inference_engine == 'Openai':
        print('---------Using OpenaiBot!-------')
        from xtuner.chat import OpenaiBot
        templates = CHAT_TEMPLATE[chat_TEMPLATE]
        bot = OpenaiBot(model_path)
        xtuner_chat_bot = BaseChat(bot, '嬛嬛', templates)

    print('init_over!')
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)


def user(user_message, history):
    return "", history + [[user_message, None]]


def get_respond(chat_history, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed):
    message = chat_history[-1][0]
    stop_words = []  # TODO
    global gen_config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_words=stop_words,
        seed=seed,
    )
    bot_message = xtuner_chat_bot.chat(message, None, gen_config)
    chat_history[-1][1] = ""
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


def predict_file(files):
    from datasets import load_dataset
    dataset = load_dataset('text', data_files=files.name)['train']
    texts = dataset['text']
    preds = xtuner_chat_bot.predict(texts=texts, generation_config=gen_config)
    dataset = dataset.add_column('response', preds)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        df = dataset.to_pandas()
        df.to_excel(tmp_file.name, 'vllm', index=False)
    return tmp_file.name


with gr.Blocks(title="XTuner Chat Board", css=CSS) as demo:

    gr.Markdown(value='''  
<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="300"/>
  <br /><br />    
                
[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)                
                ''')
    gr.HTML(
        "<h1><center>Xtuner Chat Board</h1>"
    )
    with gr.Row():
        lang = gr.Dropdown(label='language', choices=[
                           "en", "zh"], scale=1, value='en', interactive=True)
        chat_TEMPLATE = gr.Dropdown(
            label='chat_TEMPLATE', choices=chat_templates, scale=2, value='internlm_chat', interactive=True)
        # bot name
        bot_name = gr.Textbox(
            label='bot name', value='internlm', interactive=True)
        # 推理引擎
        inference_engine = gr.Dropdown(label='inference engine', choices=[
            'Huggingface', 'LMDeploy', 'Vllm', 'Openai'], value='Huggingface', interactive=True)
        init_chatbot = gr.Button(value='init_chatbot')

    model_path = gr.Textbox(
        label='model_path', value='/root/share/model_repos/internlm-chat-7b', scale=3, interactive=True)

    with gr.Accordion("Generation Parameters", open=False) as parameter_row:
        system = gr.Textbox(label='system_message',
                            value='You are a helpful assistant', scale=3, interactive=True)
        max_new_tokens = gr.Slider(
            minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)
        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.1,
                                step=0.1, interactive=True, label="Temperature",)
        repetition_penalty = gr.Slider(
            minimum=0.0, maximum=5.0, value=1.0, step=0.1, interactive=True, label="Repetition Penalty",)
        top_k = gr.Slider(minimum=1, maximum=50, value=40,
                          step=1, interactive=True, label="Top K",)
        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.75,
                          step=0.1, interactive=True, label="Top P",)
        stop_words = gr.Textbox(label='stop_words', interactive=True)
        seed = gr.Textbox(label='seed', value=0, interactive=True)

    with gr.Tab("Chat"):
        with gr.Group(visible=False) as chat_board:
            chatbot = gr.Chatbot(label='Chatbot')
            history = gr.State([])
            msg = gr.Textbox(label='Textbox')
            with gr.Row():
                ask = gr.Button('提交')
                clear = gr.ClearButton([msg, chatbot], value='Clear')
                undo = gr.Button('撤回上一条')
                regenerate = gr.Button('重新生成')
        chat_warning_info = gr.Textbox('请先完成初始化')

    with gr.Tab("批处理"):
        with gr.Group(visible=True) as porcess_board:
            file_output = gr.File(label='生成文件')
            with gr.Row():
                gr.Textbox(text_data, lines=4, label='input formate')
                gr.DataFrame(df, label='output formate')
            upload_button = gr.UploadButton(
                "Click to Upload a File", file_types=["text"])
        porcess_warning_info = gr.Textbox('请先完成初始化')

    lang.select(fn=lang_change, inputs=[lang],
                outputs=[lang, chat_TEMPLATE, model_path, inference_engine, chatbot, msg, clear, init_chatbot, bot_name])

    init_chatbot.click(fn_init_chatbot, inputs=[
                       chat_TEMPLATE, inference_engine, model_path], outputs=[chat_warning_info, chat_board, porcess_warning_info, porcess_board])

    upload_button.upload(predict_file, upload_button, file_output)

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond, [chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], chatbot)

demo.queue()
demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
