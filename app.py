from xtuner.chat import BaseChat
from xtuner.chat import CHAT_TEMPLATE
from xtuner.chat import GenerationConfig
import gradio as gr
import pandas as pd
import time
import os



chat_templates = ['internlm_chat', 'internlm2_chat', 'zephyr', 'moss_sft', 'llama2_chat', 'code_llama_chat', 'chatglm2', 'chatglm3', 'qwen_chat',
                  'baichuan_chat', 'baichuan2_chat', 'wizardlm', 'wizardcoder', 'vicuna', 'deepseek_coder', 'deepseekcoder', 'deepseek_moe', 'mistral', 'mixtral']

model_sources = ['local']

en_list = [
            ['language', 'en', 'choose what language you want to use'], ['chat_TEMPLATE','internlm_chat'],
            ['bot name','internlm'], ['inference engine', 'Huggingface', 'Select llm deployment engine'], 
            [None, 'init_chatbot'], ['model_path','/root/share/model_repos/internlm-chat-7b'], ['model_source','local'],
            ["Generation Parameters"], ['system_message', 'You are a helpful assistant'], 
            ["Top K", 40, 'At each generation step, the model considers the top K highest-ranking words in the probability distribution of the current word, and then selects one of them as the next word.'],
            ["Top P", 0.75, 'Top P defines the cumulative probability threshold of the probability mass to be considered when generating the next word. At each step, the model sorts the words in the vocabulary in descending order of probability, and then samples from the range where the cumulative probability reaches Top P'], 
            ['stop_words', None, 'Generation will be terminated when these words are generated'], ['seed', 0],
            ["Max output tokens", 512], 
            ["Temperature", 0.1, 'Controls diversity of model output'], 
            ["Repetition Penalty", 1.0, 'Reduce duplicate content in generated text'], 
            ["Basic chat"], ['Chatbot'], ['Textbox'], [None, '🚀 Submit'], [None, '🧹 Clear'], [None,'↩️ Recall last message'],
            [None, '🔁 Regenerate'], ['warning', '⚠️ Please complete initialization first'], ["File processing"], 
            ['file save path', None, 'default saved in {time}/output.xlsx'], ['output file'], 
            ['review your input', 'Please make sure your questions are line separated and saved in a text file'],
            ['review your output', 'The generated file will be saved as an excel table'], 
            [None, "Click to Upload a File"], [None, 'Generate'], 
            ['warning', '⚠️ Please complete initialization first'], 
            ["LLaVa"], 
            ["chat_template", "internlm2_chat"], ["model_path", "/root/share/model_repos/internlm2-chat-7b"],
            ["llava_template", "llava-internlm2-7b"], ["llava_path", "/root/llava/xtuner/llava-internlm2-7b"], 
            ["encoder_template", "clip-vit-large-patch14-336"], ["encoder_path", "/root/openai/clip-vit-large-patch14-336"],
            [None, "init_llava"], ["LLaVa Chatbot"], [None, None, None, "Please initialize model!"], 
            [None, '🚀 Submit'], [None, '↩️ Recall last message'], [None, '🔁 Regenerate'], [None, '🧹 Clear']
]

zh_list = [
            ['语言', 'zh', '选择语言'], ['模型模板','internlm_chat'],
            ['机器人名字','internlm'], ['推理引擎', 'Huggingface', '选择模型部署引擎'], 
            [None, '初始化模型'], ['模型路径','/root/share/model_repos/internlm-chat-7b'], ['模型来源','本地'],
            ["生成参数"], ['系统信息', 'You are a helpful assistant'], 
            ["Top K", 40, '在每一步生成中, 模型会考虑在当前词的概率分布中的前K个最高排名的词, 然后选择其中的一个词作为下一个输出.'],
            ["Top P", 0.75, 'Top P 定义了在生成下一个词时需要考虑的概率质量函数的累积概率阈值。在每一步中, 模型会以概率的降序顺序对词库中的词进行排序, 随后在累积概率达到Top P的范围内进行采样.'], 
            ['停止输出词', None, '生成会在这些词被生成出来时停止'], ['随机种子', 0],
            ["最大输出token数", 512], 
            ["温度", 0.1, '控制模型输出的强度'], 
            ["重复惩罚", 1.0, '在生成文本中减少重复内容'], 
            ["基本聊天"], ['聊天机器人'], ['文本框'], [None, '🚀 提交'], [None, '🧹 清除'], [None,'↩️ 撤回上条消息'],
            [None, '🔁 重新生成'], ['警告', '⚠️ 请先进行模型初始化'], ["文件处理"], 
            ['文件保存路径', None, '默认保存在 {time}/output.xlsx'], ['输出文件'], 
            ['审核你的输入', '请确保你的问题按行分开并且保存在一个文本文件中'],
            ['审核你的输出', '生成的文件会被保存在一个excel表格中'], 
            [None, '点击来上传文件'], [None, '生成'], 
            ['警告', '⚠️ 请先完成模型初始化'], 
            ["LLaVa"], 
            ["模型模板", "internlm2_chat"], ["模型路径", "/root/share/model_repos/internlm2-chat-7b"],
            ["llava模板", "llava-internlm2-7b"], ["llava路径", "/root/llava/xtuner/llava-internlm2-7b"], 
            ["编码器模板", "clip-vit-large-patch14-336"], ["编码器路径", "/root/openai/clip-vit-large-patch14-336"],
            [None, "初始化模型"], ["LLaVa 聊天机器人"], [None, None, None, "请初始化模型"], 
            [None, '🚀 提交'], [None, '↩️ 撤回上条信息'], [None, '🔁 重新生成'], [None, '🧹 清除']
]
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
        repetition_penalty = float(repetition_penalty),
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
        repetition_penalty = float(repetition_penalty),
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


def show_upload_file(files):
    with open(files.name, 'r') as file:
        lines = file.readlines()[:5]
    print(lines)
    return ''.join(lines)


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
        repetition_penalty = float(repetition_penalty),
        stop_words=stop_words,
        seed=seed,
    )
    preds = xtuner_chat_bot.predict(texts=texts, gen_config=gen_config)
    dataset = dataset.add_column('response', preds)
    df = dataset.to_pandas()
    if save_path == "" or save_path == None:
        folder_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        save_path = os.path.join(folder_name, 'output.xlsx')
    df.to_excel(save_path, 'vllm', index=False)
    return save_path, '\n'.join(df['response'].head(4).values)


def llava_init(llava_select, model_path, llava_path, encoder_select, encoder_path, image):
    global llava_bot
    from xtuner.chat import HFLlavaBot, LlavaChat
    template = CHAT_TEMPLATE['internlm2_chat']
    model_path = '/root/share/model_repos/internlm2-chat-7b'  # sanity assertion
    bot = HFLlavaBot(model_path, llava_path, encoder_path)
    llava_bot = LlavaChat(bot, image, chat_template=template)
    return [gr.update(placeholder="Enter text and press ENTER", interactive=True)] + [gr.update(interactive=True)] * 4


def llava_respond(chat_history, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed):
    message = chat_history[-1][0]
    stop_words = []  # TODO
    global gen_config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty = float(repetition_penalty),
        stop_words=stop_words,
        seed=seed,
    )
    bot_message = llava_bot.chat(message, gen_config=gen_config)
    chat_history[-1][1] = ""
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


def llava_tab_button_change():
    return gr.update(interactive=False)


def llava_withdraw_last_respond(chat_history):
    print(llava_bot.history)
    print(chat_history)
    llava_bot.history = llava_bot.history[:-2]
    chat_history = chat_history[:-1]
    return chat_history


def llava_regenerate_respond(chat_history, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed):
    # 删除生成的最近的内容
    chat_history[-1][1] = ""
    llava_bot.history = llava_bot.history[:-1]
    stop_words = []
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty = float(repetition_penalty),
        stop_words=stop_words,
        seed=seed,
    )
    message = chat_history[-1][0]

    bot_message = llava_bot.chat(message, gen_config=gen_config)
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


with gr.Blocks(title="XTuner Chat Board") as demo:

    gr.Markdown(value='''  
<div align="center">  
    <img src="https://s11.ax1x.com/2024/02/04/pFlcgOK.md.png" width="300"/>  
    <br /><br />          
                
[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)    </div>             
                ''')
    gr.HTML(
        "<h1><center>XTuner Chat Board</h1>"
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

    with gr.Tab("Basic chat") as basic_chat:
        with gr.Group(visible=False) as chat_board:
            chatbot = gr.Chatbot(label='Chatbot')
            history = gr.State([])
            msg = gr.Textbox(label='Textbox')
            with gr.Row():
                ask = gr.Button('🚀 Submit')
                clear = gr.Button('🧹 Clear')
                withdraw = gr.Button('↩️ Recall last message')
                regenerate = gr.Button('🔁 Regenerate')
        chat_warning_info = gr.Textbox(
            '⚠️ Please complete initialization first', label='warning')

    with gr.Tab("File processing") as file_tab:
        with gr.Group(visible=False) as process_board:
            with gr.Row():
                save_path = gr.Textbox(
                    label='file save path', info='default saved in {time}/output.xlsx')
                file_output = gr.File(label='output file')

            with gr.Row():
                input_file_content = gr.Textbox(
                    'Please make sure your questions are line separated and saved in a text file', label='review your input', max_lines=5)
                output_file_content = gr.Textbox(
                    'The generated file will be saved as an excel table', label='review your output', max_lines=5)

            with gr.Row():
                upload_button = gr.UploadButton(
                    "Click to Upload a File", file_types=["text"])
                test_but = gr.Button('Generate')

        process_warning_info = gr.Textbox(
            '⚠️ Please complete initialization first', label='warning')

    with gr.Tab("LLaVa") as llava_tab:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                llava_model = gr.Dropdown(label="chat_template", choices=[
                                           "internlm2_chat"], value="internlm2_chat", scale=1, interactive=True)
                llava_model_path = gr.Textbox(
                    label="model_path", value="/root/share/model_repos/internlm2-chat-7b", interactive=True)

                llava_select = gr.Dropdown(label="llava_template", choices=[
                                           "llava-internlm2-7b"], value="llava-internlm2-7b", scale=1, interactive=True)
                llava_path = gr.Textbox(
                    label="llava_path", value="/root/llava/xtuner/llava-internlm2-7b", interactive=True)
                encoder_select = gr.Dropdown(label="encoder_template", choices=[
                                             "clip-vit-large-patch14-336"], value="clip-vit-large-patch14-336", scale=1, interactive=True)
                encoder_path = gr.Textbox(
                    label="encoder_path", value="/root/openai/clip-vit-large-patch14-336", interactive=True)

                img_input = gr.Image(interactive=True, type='filepath')
                llava_model_init_button = gr.Button(
                    "init_llava", interactive=True)

            with gr.Column(scale=3):
                llava_chatbot = gr.Chatbot(label="LLaVa Chatbot", height=550)
                llava_history = gr.State([])
                with gr.Row():
                    llava_msg = gr.Textbox(
                        show_label=False, scale=2, placeholder="Please initialize model!", interactive=False)
                    llava_submit_button = gr.Button(
                        '🚀 Submit', scale=1, interactive=False)
                with gr.Row():
                    llava_withdraw = gr.Button(
                        '↩️ Recall last message', interactive=False)
                    llava_regenerate = gr.Button(
                        '🔁 Regenerate', interactive=False)
                    llava_clear = gr.ClearButton(
                        [llava_chatbot, llava_msg], value='🧹 Clear', interactive=False)

    components = [lang, chat_TEMPLATE, bot_name, inference_engine, init_chatbot, model_path, model_source,
                    parameter_row, system, top_k, top_p, stop_words, seed, max_new_tokens, 
                    temperature, repetition_penalty, basic_chat, chatbot, msg, ask, clear, withdraw,
                    regenerate, chat_warning_info, file_tab, save_path, file_output, input_file_content,
                    output_file_content, upload_button, test_but, process_warning_info, llava_tab,
                    llava_model, llava_model_path, llava_select, llava_path, encoder_select, encoder_path,
                    llava_model_init_button, llava_chatbot, llava_msg, llava_submit_button, llava_withdraw,
                    llava_regenerate, llava_clear]

    def lang_change(lang):
        com_len = len(components)
        return_list = []
        if lang == "en":
            for i in range(com_len):
                com = components[i]
                com_list = en_list[i]
                if isinstance(com, gr.Button) or isinstance(com, gr.UploadButton) or isinstance(com, gr.ClearButton):
                    return_list += [gr.update(value=com_list[1])]
                elif len(com_list) == 1:
                    return_list += [gr.update(label=com_list[0])]
                elif len(com_list) == 2:
                    return_list += [gr.update(label=com_list[0], value=com_list[1])]
                elif len(com_list) == 3:
                    return_list += [gr.update(label=com_list[0], value=com_list[1], info=com_list[2])]
                elif len(com_list) == 4:
                    return_list += [gr.update(label=com_list[0], value=com_list[1], info=com_list[2], placeholder=com_list[3])]

        elif lang == "zh":
            for i in range(com_len):
                com = components[i]
                com_list = zh_list[i]
                if isinstance(com, gr.Button) or isinstance(com, gr.UploadButton) or isinstance(com, gr.ClearButton):
                    return_list += [gr.update(value=com_list[1])]
                elif len(com_list) == 1:
                    return_list += [gr.update(label=com_list[0])]
                elif len(com_list) == 2:
                    return_list += [gr.update(label=com_list[0], value=com_list[1])]
                elif len(com_list) == 3:
                    return_list += [gr.update(label=com_list[0], value=com_list[1], info=com_list[2])]
                elif len(com_list) == 4:
                    return_list += [gr.update(label=com_list[0], value=com_list[1], info=com_list[2], placeholder=com_list[3])]

        return return_list

    lang.select(fn=lang_change, inputs=lang, outputs=components, queue=False)

    init_chatbot.click(fn_init_chatbot, inputs=[
                       chat_TEMPLATE, inference_engine, model_path], outputs=[chat_warning_info, chat_board, process_warning_info, process_board])

    upload_button.upload(show_upload_file, inputs=[upload_button], outputs=[input_file_content]).then(predict_file, inputs=[upload_button, max_new_tokens, temperature,
                                                                                                                            repetition_penalty, top_k, top_p, stop_words, seed, save_path], outputs=[file_output, output_file_content])

    ask.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond, [chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], chatbot)

    clear.click(clear_respond, outputs=[msg, chatbot])

    withdraw.click(withdraw_last_respond, inputs=[chatbot], outputs=[chatbot])

    regenerate.click(regenerate_respond, inputs=[
                     chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], outputs=[chatbot])

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond, [chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], chatbot)

    # llava events
    # llava_tab.select(
    #    llava_tab_button_change, outputs=init_chatbot
    # )
    llava_model_init_button.click(
        llava_init, inputs=[llava_select, model_path,
                            llava_path, encoder_select, encoder_path, img_input],
        outputs=[llava_msg, llava_submit_button,
                 llava_withdraw, llava_regenerate, llava_clear]
    )
    llava_msg.submit(user, [llava_msg, llava_chatbot], [llava_msg, llava_chatbot], queue=False).then(
        llava_respond, [llava_chatbot, max_new_tokens, temperature,
                        repetition_penalty, top_k, top_p, stop_words, seed], llava_chatbot
    )
    llava_submit_button.click(user, [llava_msg, llava_chatbot], [llava_msg, llava_chatbot], queue=False).then(
        llava_respond, [llava_chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], llava_chatbot)

    llava_withdraw.click(llava_withdraw_last_respond, inputs=[
                         llava_chatbot], outputs=[llava_chatbot])

    llava_regenerate.click(llava_regenerate_respond, inputs=[
        llava_chatbot, max_new_tokens, temperature, repetition_penalty, top_k, top_p, stop_words, seed], outputs=[llava_chatbot])


demo.queue()
demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
