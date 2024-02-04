# XTuner Chat GUI
<div align="center">  
    <img src="https://s11.ax1x.com/2024/02/04/pFlcgOK.md.png" width="300"/>  
    <br /><br />          
</div>   
欢迎使用 XTuner Chat GUI，这是您快速部署本地模型的理想解决方案，利用了 XTuner 后端框架。这个图形用户界面（GUI）旨在简化本地模型推理部署，支持多个后端推理引擎。XTuner Chat GUI 不仅是一个强大的本地模型部署工具，还提供了批量处理本地文件推理请求以及支持多模态 Llava Chat 功能等特色功能。

## 目录
- [介绍](#介绍)
- [特点](#特点)
- [入门指南](#入门指南)
- [贡献](#贡献)
- [许可证](#许可证)

## 介绍

XTuner Chat GUI 是建立在 XTuner 后端框架之上的用户友好界面。它通过支持多个后端推理引擎，提供对本地模型的快速而高效的部署。除了支持批量处理和多模态 Llava Chat 等功能，XTuner Chat GUI 旨在满足您多样化的推理需求。

## 特点

1. **快速部署：** 利用 XTuner 后端框架，快速高效地部署本地模型。
2. **后端引擎支持：** XTuner Chat GUI 支持多个后端推理引擎，提供灵活性和兼容性。
3. **批量处理：** 通过批量处理本地文件中的推理请求，轻松提高大规模任务的效率。
4. **多模态 Llava Chat：** 支持多模态 Llava Chat 功能，实现多样化的交互方式。

## 入门指南

按照以下步骤开始使用 XTuner Chat GUI：

1. **安装：** 克隆存储库并安装所需的依赖项。
    ```bash
    git clone https://github.com/limafang/Xtuner-GUI
    cd Xtuner-GUI
    pip install -r requirements.txt
    ```

2. **运行 XTuner Chat GUI:**
    ```bash
    python web_demo.py
    ```

3. **访问界面：** 打开您的 Web 浏览器，导航至 [http://localhost:7086](http://localhost:7086) 以访问 XTuner Chat GUI。



## 使用

一旦 XTuner Chat GUI 运行，您可以探索各种功能：

- **本地模型部署：** 上传并部署本地模型，利用 XTuner 后端。
- **后端引擎选择：** 从多个支持的后端推理引擎中选择。
- **批量处理：** 通过上传本地文件批量处理推理请求。
- **Llava Chat：** 通过 Llava Chat 功能体验多模态交互。

有关详细的使用说明，请参阅 [用户指南](usage.md)。



## TODO

- [ ] 支持 LMDeploy，Vllm，Openai等更多推理引擎

- [ ] 支持 本地启动 web server 服务，支持 openai 服务对接

- [ ] 支持更多 llava 模型

## 贡献

感谢 @pppppM 的技术支持！

感谢 @limafang @PrinceRay7 的贡献！

我们欢迎社区的贡献。如果您有想法、错误报告或功能请求，请提出问题或提交拉取请求。

## 许可证

XTuner Chat GUI 根据 [MIT 许可证](LICENSE) 授权。请根据许可证的条款自由使用、修改和分发。

感谢选择 XTuner Chat GUI。祝您愉快部署！