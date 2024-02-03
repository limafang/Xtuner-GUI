from xtuner.chat import BaseChat
from xtuner.chat import CHAT_TEMPLATE
from xtuner.chat import GenerationConfig
import gradio as gr
import pandas as pd
import time
import os
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

model_sources = ['local']


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

# outputs=[lang, chat_TEMPLATE, model_path, inference_engine, chatbot, msg, clear, init_chatbot, bot_name])


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
    bot_message = xtuner_chat_bot.chat(message, gen_config=gen_config)
    chat_history[-1][1] = ""
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


def regenerate_respond(chat_history, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed):
    # 删除生成的最近的内容
    chat_history[-1][1] = ""
    xtuner_chat_bot.history = xtuner_chat_bot.history[:-1]
    stop_words = []
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_words=stop_words,
        seed=seed,
    )
    message = chat_history[-1][0]

    bot_message = xtuner_chat_bot.chat(message, gen_config=gen_config)
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


def clear_respond():
    xtuner_chat_bot.reset_history()
    return "", ""


def withdraw_last_respond(chat_history):
    print(xtuner_chat_bot.history)
    print(chat_history)
    xtuner_chat_bot.history = xtuner_chat_bot.history[:-2]
    chat_history = chat_history[:-1]
    return chat_history


def predict_file(files, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed, save_path):
    from datasets import load_dataset
    dataset = load_dataset('text', data_files=files.name)['train']
    texts = dataset['text']
    stop_words = []
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_words=stop_words,
        seed=seed,
    )
    preds = xtuner_chat_bot.predict(texts=texts, generation_config=gen_config)
    dataset = dataset.add_column('response', preds)
    df = dataset.to_pandas()
    if save_path == "" or save_path == None:
        folder_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        save_path = os.path.join(folder_name, 'output.xlsx')
    df.to_excel(save_path, 'vllm', index=False)
    return save_path


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
                           "en", "zh"], scale=1, value='en', interactive=True, info='choose what language you want to use')
        chat_TEMPLATE = gr.Dropdown(
            label='chat_TEMPLATE', choices=chat_templates, scale=2, value='internlm_chat', interactive=True)
        # bot name
        bot_name = gr.Textbox(
            label='bot name', value='internlm', interactive=True)
        # 推理引擎
        inference_engine = gr.Dropdown(label='inference engine', choices=[
            'Huggingface', 'LMDeploy', 'Vllm', 'Openai'], value='Huggingface', interactive=True, info='Select llm deployment engine')
        init_chatbot = gr.Button(value='init_chatbot')

    with gr.Row():
        model_path = gr.Textbox(
            label='model_path', value='/root/share/model_repos/internlm-chat-7b', scale=3, interactive=True)
        model_source = gr.Dropdown(
            label='model_source', choices=model_sources, value='local')

    with gr.Accordion("Generation Parameters", open=False) as parameter_row:
        system = gr.Textbox(label='system_message',
                            value='You are a helpful assistant', scale=3, interactive=True)
        top_k = gr.Slider(minimum=1, maximum=50, value=40,
                          step=1, interactive=True, label="Top K", info='At each generation step, the model considers the top K highest-ranking words in the probability distribution of the current word, and then selects one of them as the next word.')
        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.75,
                          step=0.1, interactive=True, label="Top P", info='Top P defines the cumulative probability threshold of the probability mass to be considered when generating the next word. At each step, the model sorts the words in the vocabulary in descending order of probability, and then samples from the range where the cumulative probability reaches Top P')
        stop_words = gr.Textbox(label='stop_words', interactive=True,
                                info='Generation will be terminated when these words are generated')
        seed = gr.Textbox(label='seed', value=0, interactive=True)
    with gr.Row():
        max_new_tokens = gr.Slider(
            minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)
        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.1,
                                step=0.1, interactive=True, label="Temperature", info='Controls diversity of model output')
        repetition_penalty = gr.Slider(
            minimum=0.0, maximum=5.0, value=1.0, step=0.1, interactive=True, label="Repetition Penalty", info='Reduce duplicate content in generated text')
    with gr.Tab("Basic chat"):
        with gr.Group(visible=False) as chat_board:
            chatbot = gr.Chatbot(label='Chatbot')
            history = gr.State([])
            msg = gr.Textbox(label='Textbox')
            with gr.Row():
                ask = gr.Button('🚀 Submmit')
                clear = gr.Button('🧹 Clear')
                withdraw = gr.Button('↩️ Recall last message')
                regenerate = gr.Button('🔁 Regenerate')
        chat_warning_info = gr.Textbox(
            '⚠️ Please complete initialization first')

    with gr.Tab("File processing"):
        with gr.Group(visible=False) as porcess_board:
            with gr.Row():
                file_output = gr.File(label='output file')
                gr.DataFrame(df, label='review your output')
            save_path = gr.Textbox(
                label='file save path', info='default saved in {time}/output.xlsx')
            with gr.Row():
                gr.Textbox(text_data, lines=4, label='input example')
                gr.DataFrame(df, label='output example')
            upload_button = gr.UploadButton(
                "Click to Upload a File", file_types=["text"])
        porcess_warning_info = gr.Textbox(
            '⚠️ Please complete initialization first')

    lang.select(fn=lang_change, inputs=[lang],
                outputs=[lang, chat_TEMPLATE, model_path, inference_engine, chatbot, msg, clear, init_chatbot, bot_name])

    init_chatbot.click(fn_init_chatbot, inputs=[
                       chat_TEMPLATE, inference_engine, model_path], outputs=[chat_warning_info, chat_board, porcess_warning_info, porcess_board])

    upload_button.upload(predict_file, inputs=[upload_button, max_new_tokens, temperature,
                         repetition_penalty, top_k, top_p, stop_words, seed, save_path], outputs=[file_output])

    ask.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond, [chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], chatbot)

    clear.click(clear_respond, outputs=[msg, chatbot])

    withdraw.click(withdraw_last_respond, inputs=[chatbot], outputs=[chatbot])

    regenerate.click(regenerate_respond, inputs=[
                     chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], outputs=[chatbot])

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond, [chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], chatbot)

demo.queue()
demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
