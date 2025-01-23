[[英文主页]](README.md) | [[DJ-Cookbook]](#dj-cookbook) | [[算子池]](docs/Operators.md) | [[API]](https://modelscope.github.io/data-juicer) | [[Awesome LLM Data]](docs/awesome_llm_data.md)

# Data Processing for and with Foundation Models

 <img src="https://img.alicdn.com/imgextra/i1/O1CN01fUfM5A1vPclzPQ6VI_!!6000000006165-0-tps-1792-1024.jpg" width = "533" height = "300" alt="Data-Juicer"/>

![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)

[![DataModality](https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg)](#dj-cookbook)
[![Usage](https://img.shields.io/badge/Usage-Cleaning,Synthesis,Analysis-FFD21E.svg)](#dj-cookbook)
[![ModelScope- Demos](https://img.shields.io/badge/ModelScope-Demos-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/studios?name=Data-Jiucer&page=1&sort=latest&type=1)
[![HuggingFace- Demos](https://img.shields.io/badge/🤗HuggingFace-Demos-4e29ff.svg)](https://huggingface.co/spaces?&search=datajuicer)

[![Document_List](https://img.shields.io/badge/Doc-DJ_Cookbook-blue?logo=Markdown)](#dj-cookbook)
[![文档列表](https://img.shields.io/badge/文档-DJ指南-blue?logo=Markdown)](README_ZH.md#dj-cookbook)
[![算子池](https://img.shields.io/badge/文档-算子池-blue?logo=Markdown)](docs/Operators.md)




Data-Juicer 是一个一站式系统，面向大模型的文本及多模态数据处理。我们提供了一个基于 JupyterLab 的 [Playground](http://8.138.149.181/)，您可以从浏览器中在线试用 Data-Juicer。 如果Data-Juicer对您的研发有帮助，请支持加星（自动订阅我们的新发布）、以及引用我们的[工作](#参考文献) 。



<div id="table" align="center"></div>

目录
===
- [新消息](#新消息)
- [为什么选择 Data-Juicer？](#为什么选择-data-juicer)
- [DJ-Cookbook](#dj-cookbook)
  - [资源合集](#资源合集)
  - [编写Data-Juicer (DJ) 代码](#编写data-juicer-dj-代码)
  - [用例与数据菜谱](#用例与数据菜谱)
  - [交互类示例](#交互类示例)
- [安装](#安装)
  - [前置条件](#前置条件)
  - [从源码安装](#从源码安装)
  - [使用 pip 安装](#使用-pip-安装)
  - [使用 Docker 安装](#使用-docker-安装)
  - [安装校验](#安装校验)
  - [使用视频相关算子](#使用视频相关算子)
- [快速上手](#快速上手)
  - [数据处理](#数据处理)
  - [分布式数据处理](#分布式数据处理)
  - [数据分析](#数据分析)
  - [数据可视化](#数据可视化)
  - [构建配置文件](#构建配置文件)
  - [沙盒实验室](#沙盒实验室)
  - [预处理原始数据（可选）](#预处理原始数据可选)
  - [对于 Docker 用户](#对于-docker-用户)
- [开源协议](#开源协议)
- [贡献](#贡献)
- [致谢](#致谢)
- [参考文献](#参考文献)


## 为什么选择 Data-Juicer？

<img src="https://img.alicdn.com/imgextra/i2/O1CN01EteoQ31taUweAW1UE_!!6000000005918-2-tps-4034-4146.png" align="center" width="600" />

- **系统化和可重用**：
系统化地为用户提供 100 多个核心 [算子](docs/Operators.md) 和 50 多个可重用的数据菜谱和
专用工具套件，旨在解耦于特定的多模态 LLM 数据集和处理管道运行。支持预训练、后训练、英语、中文等场景中的数据分析、清洗和合成。

- **易用、可扩展**：
简洁灵活，提供快速[入门指南](#快速上手)和包含丰富使用示例的[DJ-Cookbook](#dj-cookbook)。您可以灵活实现自己的OP，[自定义](docs/DeveloperGuide_ZH.md)数据处理工作流。

- **高效、稳定**：提供性能优化的[并行数据处理能力](docs/Distributed_ZH.md)（Aliyun-PAI\Ray\CUDA\OP Fusion），
更快、更少资源消耗，基于大规模生产环境打磨。

- **效果验证、沙盒**：支持数据模型协同开发，通过[沙盒实验室](docs/Sandbox-ZH.md)实现快速迭代，提供反馈循环、可视化等功能，让您更好地理解和改进数据和模型。已经有许多基于 DJ 衍生的数据菜谱和模型经过了效用验证，譬如在预训练、文生视频、图文生成等场景。
![Data-in-the-loop](https://img.alicdn.com/imgextra/i2/O1CN017U7Zz31Y7XtCJ5GOz_!!6000000003012-0-tps-3640-1567.jpg)

## DJ-Cookbook
### 资源合集
- [KDD'24 相关教程](https://modelscope.github.io/data-juicer/_static/tutorial_kdd24.html)
- [Awesome LLM-Data](docs/awesome_llm_data.md)
- [“坏”数据展览](docs/BadDataExhibition_ZH.md)

### 编写Data-Juicer (DJ) 代码
- 基础
  - [DJ概览](README_ZH.md)
  - [快速上手](#快速上手)
  - [配置](docs/RecipeGallery_ZH.md)
  - [数据格式转换](tools/fmt_conversion/README_ZH.md)
- 信息速查
  - [算子库](docs/Operators.md)
  - [API参考](https://modelscope.github.io/data-juicer/)
- 进阶
  - [开发者指南](docs/DeveloperGuide_ZH.md)
  - [预处理工具](tools/preprocess/README_ZH.md)
  - [后处理工具](tools/postprocess/README_ZH.md)
  - [沙盒](docs/Sandbox-ZH.md)
  - [质量分类器](tools/quality_classifier/README_ZH.md)
  - [自动评估](tools/evaluator/README_ZH.md)
  - [第三方集成](thirdparty/LLM_ecosystems/README_ZH.md)

### 用例与数据菜谱
* [数据菜谱Gallery](docs/RecipeGallery.md)
  - Data-Juicer 最小示例配方
  - 复现开源文本数据集
  - 改进开源文本预训练数据集
  - 改进开源文本后处理数据集
  - 合成对比学习图像文本数据集
  - 改进开源图像文本数据集
  - 视频数据的基本示例菜谱
  - 合成以人为中心的视频评测集
  - 改进现有的开源视频数据集
* Data-Juicer相关竞赛
  - [Better Synth](https://tianchi.aliyun.com/competition/entrance/532251)，在DJ-沙盒实验室和多模态大模型上，探索大模型合成数据对图像理解能力的影响
  - [Modelscope-Sora挑战赛](https://tianchi.aliyun.com/competition/entrance/532219)，基于Data-Juicer和[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)框架，调优文本-视频数据集，在类SORA小模型上训练以生成更好的视频
  - [Better Mixture](https://tianchi.aliyun.com/competition/entrance/532174)，针对指定多个候选数据集，仅调整数据混合和采样策略
  - FT-Data Ranker ([1B Track](https://tianchi.aliyun.com/competition/entrance/532157)、 [7B Track](https://tianchi.aliyun.com/competition/entrance/532158))，针对指定候选数据集，仅调整数据过滤和增强策略
  - [可图Kolors-LoRA风格故事挑战赛](https://tianchi.aliyun.com/competition/entrance/532254)，基于Data-Juicer和[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)框架，探索Difussion模型微调
* [DJ-SORA](docs/DJ_SORA_ZH.md)
* 基于Data-Juicer和[AgentScope](https://github.com/modelscope/agentscope)框架，通过[智能体调用DJ Filters](./demos/api_service/react_data_filter_process.ipynb)和[调用DJ Mappers](./demos/api_service/react_data_mapper_process.ipynb)
  


### 交互类示例
* Data-Juicer 介绍 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
* 数据可视化:
  * 基础指标统计 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  * 词汇多样性 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  * 算子洞察（单OP） [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_insight)]
  * 算子效果（多OP） [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
* 数据处理:
  * 科学文献 (例如 [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  * 编程代码 (例如 [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  * 中文指令数据 (例如 [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
* 工具池:
  * 按语言分割数据集 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  * CommonCrawl 质量分类器 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  * 基于 [HELM](https://github.com/stanford-crfm/helm) 的自动评测 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  * 数据采样及混合 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
* 数据处理回路 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]


## 安装

### 前置条件

* 推荐 Python>=3.9,<=3.10
* gcc >= 5 (at least C++14 support)


### 从源码安装

* 运行以下命令以安装 `data_juicer` 可编辑模式的最新基础版本

```shell
cd <path_to_data_juicer>
pip install -v -e .
```

* 部分算子功能依赖于较大的或者平台兼容性不是很好的第三方库，因此用户可按需额外安装可选的依赖项:

```shell
cd <path_to_data_juicer>
pip install -v -e .  # 安装最小依赖，支持基础功能
pip install -v -e .[tools] # 安装部分工具库的依赖
```

依赖选项如下表所示:

| 标签               | 描述                           |
|------------------|------------------------------|
| `.` 或者 `.[mini]` | 安装支持 Data-Juicer 基础功能的最小依赖项  |
| `.[all]`         | 安装除了沙盒实验以外的所有依赖项  |
| `.[sci]`         | 安装所有算子的全量依赖                  |
| `.[dist]`        | 安装以分布式方式进行数据处理的依赖（实验性功能）     |
| `.[dev]`         | 安装作为贡献者开发 Data-Juicer 所需的依赖项 |
| `.[tools]`       | 安装专用工具库（如质量分类器）所需的依赖项        |
| `.[sandbox]`     | 安装沙盒实验室的基础依赖                 |

* 只安装部分算子依赖

随着OP数量的增长，所有OP的依赖变得很重。为此，我们提供了两个替代的、更轻量的选项，作为使用命令`pip install -v -e .[sci]`安装所有依赖的替代：

  * 自动最小依赖安装：在执行Data-Juicer的过程中，将自动安装最小依赖。也就是说你可以直接执行，但这种方式可能会导致一些依赖冲突。

  * 手动最小依赖安装：可以通过如下指令手动安装适合特定执行配置的最小依赖：
    ```shell
    # 适用于从源码安装
    python tools/dj_install.py --config path_to_your_data-juicer_config_file
    
    # 使用命令行工具
    dj-install --config path_to_your_data-juicer_config_file
    ```

### 使用 pip 安装

* 运行以下命令用 `pip` 安装 `data_juicer` 的最新发布版本：

```shell
pip install py-data-juicer
```

* **注意**：
  * 使用这种方法安装时，只有`data_juicer`中的基础的 API 和2个基础工具
    （数据[处理](#数据处理)与[分析](#数据分析)）可以使用。如需更定制化地使用完整功能，建议[从源码进行安装](#从源码安装)。
  * pypi 的发布版本较源码的最新版本有一定的滞后性，如需要随时跟进 `data_juicer` 的最新功能支持，建议[从源码进行安装](#从源码安装)。

### 使用 Docker 安装

- 您可以选择
  - 从DockerHub直接拉取我们的预置镜像:
    ```shell
    docker pull datajuicer/data-juicer:<version_tag>
    ```
  - 或者运行如下命令用我们提供的 [Dockerfile](Dockerfile) 来构建包括最新版本的 `data-juicer` 的 docker 镜像：

    ```shell
    docker build -t datajuicer/data-juicer:<version_tag> .
    ```

  - `<version_tag>`的格式类似于`v0.2.0`，与发布（Release）的版本号相同。

### 安装校验

```python
import data_juicer as dj
print(dj.__version__)
```

### 使用视频相关算子

在使用视频相关算子之前，应该安装 **FFmpeg** 并确保其可通过 $PATH 环境变量访问。

你可以使用包管理器安装 FFmpeg（例如，在 Debian/Ubuntu 上使用 sudo apt install ffmpeg，在 OS X 上使用 brew install ffmpeg），或访问[官方FFmpeg链接](https://ffmpeg.org/download.html)。

随后在终端运行 ffmpeg 命令检查环境是否设置正确。


<p align="right"><a href="#table">🔼 back to index</a></p>

## 快速上手

### 数据处理

* 以配置文件路径作为参数来运行 `process_data.py` 或者 `dj-process` 命令行工具来处理数据集。

```shell
# 适用于从源码安装
python tools/process_data.py --config configs/demo/process.yaml

# 使用命令行工具
dj-process --config configs/demo/process.yaml
```

* **注意**：使用未保存在本地的第三方模型或资源的算子第一次运行可能会很慢，因为这些算子需要将相应的资源下载到缓存目录中。默认的下载缓存目录为`~/.cache/data_juicer`。您可通过设置 shell 环境变量 `DATA_JUICER_CACHE_HOME` 更改缓存目录位置，您也可以通过同样的方式更改 `DATA_JUICER_MODELS_CACHE` 或 `DATA_JUICER_ASSETS_CACHE` 来分别修改模型缓存或资源缓存目录:

* **注意**：对于使用了第三方模型的算子，在填写config文件时需要去声明其对应的`mem_required`（可以参考`config_all.yaml`文件中的设置）。Data-Juicer在运行过程中会根据内存情况和算子模型所需的memory大小来控制对应的进程数，以达成更好的数据处理的性能效率。而在使用CUDA环境运行时，如果不正确的声明算子的`mem_required`情况，则有可能导致CUDA Out of Memory。

```shell
# 缓存主目录
export DATA_JUICER_CACHE_HOME="/path/to/another/directory"
# 模型缓存目录
export DATA_JUICER_MODELS_CACHE="/path/to/another/directory/models"
# 资源缓存目录
export DATA_JUICER_ASSETS_CACHE="/path/to/another/directory/assets"
```

- **灵活的编程接口：**
我们提供了各种层次的简单编程接口，以供用户选择：
```python
# ... init op & dataset ...

# 链式调用风格，支持单算子或算子列表
dataset = dataset.process(op)
dataset = dataset.process([op1, op2])
# 函数式编程风格，方便快速集成或脚本原型迭代
dataset = op(dataset)
dataset = op.run(dataset)
```

### 分布式数据处理

Data-Juicer 现在基于[RAY](https://www.ray.io/)实现了多机分布式数据处理。
对应Demo可以通过如下命令运行：

```shell

# 运行文字数据处理
python tools/process_data.py --config ./demos/process_on_ray/configs/demo.yaml

# 运行视频数据处理
python tools/process_data.py --config ./demos/process_video_on_ray/configs/demo.yaml

```

 - 如果需要在多机上使用RAY执行数据处理，需要确保所有节点都可以访问对应的数据路径，即将对应的数据路径挂载在共享文件系统（如NAS）中。
 - RAY 模式下的去重算子与单机版本不同，所有 RAY 模式下的去重算子名称都以 `ray` 作为前缀，例如 `ray_video_deduplicator` 和 `ray_document_deduplicator`。
 - 更多细节请参考[分布式处理文档](docs/Distributed_ZH.md)。

> 用户也可以不使用 RAY，拆分数据集后使用 [Slurm](https://slurm.schedmd.com/) 在集群上运行，此时使用不包含 RAY 的原版 Data-Juicer 即可。
> [阿里云 PAI-DLC](https://www.aliyun.com/activity/bigdata/pai-dlc) 支持 RAY 框架、Slurm 框架等，用户可以直接在DLC集群上创建 RAY 作业 和 Slurm 作业。

### 数据分析

- 以配置文件路径为参数运行 `analyze_data.py` 或者 `dj-analyze` 命令行工具来分析数据集。

```shell
# 适用于从源码安装
python tools/analyze_data.py --config configs/demo/analyzer.yaml

# 使用命令行工具
dj-analyze --config configs/demo/analyzer.yaml

# 你也可以使用"自动"模式来避免写一个新的数据菜谱。它会使用全部可产出统计信息的 Filter 来分析
# 你的数据集的一小部分（如1000条样本，可通过 `auto_num` 参数指定）
dj-analyze --auto --dataset_path xx.jsonl [--auto_num 1000]
```

* **注意**：Analyzer 只用于能在 stats 字段里产出统计信息的 Filter 算子和能在 meta 字段里产出 tags 或类别标签的其他算子。除此之外的其他的算子会在分析过程中被忽略。我们使用以下两种注册器来装饰相关的算子：
  * `NON_STATS_FILTERS`：装饰那些**不能**产出任何统计信息的 Filter 算子。
  * `TAGGING_OPS`：装饰那些能在 meta 字段中产出 tags 或类别标签的算子。

### 数据可视化

* 运行 `app.py` 来在浏览器中可视化您的数据集。
* **注意**：只可用于从源码安装的方法。

```shell
streamlit run app.py
```




### 构建配置文件

* 配置文件包含一系列全局参数和用于数据处理的算子列表。您需要设置:
  * 全局参数：输入/输出 数据集路径，worker 进程数量等。
  * 算子列表：列出用于处理数据集的算子及其参数。
* 您可以通过如下方式构建自己的配置文件:
  * ➖：修改我们的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)。该文件包含了**所有**算子以及算子对应的默认参数。您只需要**移除**不需要的算子并重新设置部分算子的参数即可。
  * ➕：从头开始构建自己的配置文件。您可以参考我们提供的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)，[算子文档](docs/Operators.md)，以及 [开发者指南](docs/DeveloperGuide_ZH.md#构建自己的算子).
  * 除了使用 yaml 文件外，您还可以在命令行上指定一个或多个参数，这些参数将覆盖 yaml 文件中的值。

```shell
python xxx.py --config configs/demo/process.yaml --language_id_score_filter.lang=en
```

* 基础的配置项格式及定义如下图所示

  ![基础配置项格式及定义样例](https://img.alicdn.com/imgextra/i4/O1CN01xPtU0t1YOwsZyuqCx_!!6000000003050-0-tps-1692-879.jpg "基础配置文件样例")

### 沙盒实验室

数据沙盒实验室 (DJ-Sandbox) 为用户提供了持续生产数据菜谱的最佳实践，其具有低开销、可迁移、有指导性等特点。
- 用户在沙盒中可以基于一些小规模数据集、模型对数据菜谱进行快速实验、迭代、优化，再迁移到更大尺度上，大规模生产高质量数据以服务大模型。
- 用户在沙盒中，除了Data-Juicer基础的数据优化与数据菜谱微调功能外，还可以便捷地使用数据洞察与分析、沙盒模型训练与评测、基于数据和模型反馈优化数据菜谱等可配置组件，共同组成完整的一站式数据-模型研发流水线。

沙盒默认通过如下命令运行，更多介绍和细节请参阅[沙盒文档](docs/Sandbox-ZH.md).
```shell
python tools/sandbox_starter.py --config configs/demo/sandbox/sandbox.yaml
```



### 预处理原始数据（可选）

* 我们的 Formatter 目前支持一些常见的输入数据集格式：
  * 单个文件中包含多个样本：jsonl/json、parquet、csv/tsv 等。
  * 单个文件中包含单个样本：txt、code、docx、pdf 等。
* 但来自不同源的数据是复杂和多样化的，例如:
  * [从 S3 下载的 arXiv 原始数据](https://info.arxiv.org/help/bulk_data_s3.html) 包括数千个 tar 文件以及更多的 gzip 文件，并且所需的 tex 文件在 gzip 文件中，很难直接获取。
  * 一些爬取的数据包含不同类型的文件（pdf、html、docx 等），并且很难提取额外的信息，例如表格、图表等。
* Data-Juicer 不可能处理所有类型的数据，欢迎提 Issues/PRs，贡献对新数据类型的处理能力！
* 因此我们在 [`tools/preprocess`](tools/preprocess) 中提供了一些**常见的预处理工具**，用于预处理这些类型各异的数据。
  * 欢迎您为社区贡献新的预处理工具。
  * 我们**强烈建议**将复杂的数据预处理为 jsonl 或 parquet 文件。

### 对于 Docker 用户

- 如果您构建或者拉取了 `data-juicer` 的 docker 镜像，您可以使用这个 docker 镜像来运行上面提到的这些命令或者工具。
- 直接运行：

```shell
# 直接运行数据处理
docker run --rm \  # 在处理结束后将容器移除
  --privileged \
  --shm-size 256g \
  --network host \
  --gpus all \
  --name dj \  # 容器名称
  -v <host_data_path>:<image_data_path> \  # 将本地的数据或者配置目录挂载到容器中
  -v ~/.cache/:/root/.cache/ \  # 将 cache 目录挂载到容器以复用 cache 和模型资源（推荐）
  datajuicer/data-juicer:<version_tag> \  # 运行的镜像
  dj-process --config /path/to/config.yaml  # 类似的数据处理命令
```

- 或者您可以进入正在运行的容器，然后在可编辑模式下运行命令：

```shell
# 启动容器
docker run -dit \  # 在后台启动容器
  --privileged \
  --shm-size 256g \
  --network host \
  --gpus all \
  --rm \
  --name dj \
  -v <host_data_path>:<image_data_path> \
  -v ~/.cache/:/root/.cache/ \
  datajuicer/data-juicer:latest /bin/bash

# 进入这个容器，然后您可以在编辑模式下使用 data-juicer
docker exec -it <container_id> bash
```


<p align="right"><a href="#table">🔼 back to index</a></p>

## 开源协议

Data-Juicer 在 Apache License 2.0 协议下发布。

## 贡献

大模型是一个高速发展的领域，我们非常欢迎贡献新功能、修复漏洞以及文档改善。请参考[开发者指南](docs/DeveloperGuide_ZH.md)。


## 致谢

Data-Juicer被许多大模型相关产品和研究工作所使用，例如阿里巴巴通义和阿里云人工智能平台 (PAI) 之上的工业界场景。 我们期待更多您的体验反馈、建议和合作共建！


Data-Juicer 感谢社区[贡献者](https://github.com/modelscope/data-juicer/graphs/contributors) 和相关的先驱开源项目，譬如[Huggingface-Datasets](https://github.com/huggingface/datasets), [Bloom](https://huggingface.co/bigscience/bloom), [RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1), [Arrow](https://github.com/apache/arrow), [Ray](https://github.com/ray-project/ray), ....


<p align="right"><a href="#table">🔼 back to index</a></p>