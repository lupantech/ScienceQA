# ScienceQA: Science Question Answering

![VQA](https://img.shields.io/badge/Task-VQA-red) 
![Science Problems](https://img.shields.io/badge/Task-Scientific_Reasoning-red) 
![Open Domain](https://img.shields.io/badge/Task-Open--Domain-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![ScienceQA](https://img.shields.io/badge/Dataset-ScienceQA-blue) 
![Chain-of-Thought](https://img.shields.io/badge/Model-Chain_of_Thought-green) 
![GPT-3](https://img.shields.io/badge/Model-GPT--3-green) 
![ChatGPT](https://img.shields.io/badge/Model-ChatGPT-green) 
![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![LLMs](https://img.shields.io/badge/Model-LLMs-green)

Data and code for NeurIPS 2022 Paper "[Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](http://lupantech.github.io/papers/neurips22_scienceqa.pdf)".

For more details, please refer to the project page with dataset exploration and visualization tools: https://scienceqa.github.io.

:bell: If you have any questions or suggestions, please don't hesitate to let us know. You can directly email [Pan Lu](https://lupantech.github.io/) at UCLA using the email address lupantech@gmail.com, comment on the [Twitter](https://twitter.com/lupantech/status/1570828580346802178), or post an issue on this repository.



## 💥 News 💥

- **[2022.09.20]** Our work is featured in [Deep AI](https://deepai.org/publication/learn-to-explain-multimodal-reasoning-via-thought-chains-for-science-question-answering).

- **[2022.09.22]** Our work is accepted to [NeurIPS 2022](https://nips.cc/Conferences/2022). :star:

- **[2022.11]** Our work is now included at [Paper with Code](https://paperswithcode.com/dataset/scienceqa).

- **[2022.11.20]** Our work is covered by [Geek Culture | Medium](https://medium.com/geekculture/neurips-2022-the-first-multi-modal-science-question-answering-science-qa-dataset-with-detailed-45b8d628b301).

- **[2022.11.29]** Our work gives an poster presentation by Pan Lu at [NeurIPS 2022](https://nips.cc/Conferences/2022).

- **[2023.02.02]** The ScienceQA dataset has served as the primary benchmark for the new generation of multimodal reasoning systems, [Multimodal-CoT](https://github.com/amazon-science/mm-cot), developed by [Amazon Science](https://www.amazon.science/).

- **[2023.02.24]** The ScienceQA dataset is now included at [HuggingFace Datasets](https://huggingface.co/datasets/derek-thomas/ScienceQA). :star:

- **[2023.02.05]** Our work is covered by [MarkTechPost](https://www.marktechpost.com/2023/02/05/a-new-artificial-intelligence-research-proposes-multimodal-chain-of-thought-reasoning-in-language-models-that-outperforms-gpt-3-5-by-16-75-17-%E2%86%92-91-68-on-scienceqa/).

- **[2023.02.13]** Our work gives an oral presentation by Pan Lu at [AAAI 2023 KnowledgeNLP Workshop](https://knowledge-nlp.github.io/aaai2023/publications.html).

- **[2023.03.28]** The ScienceQA dataset has served as the primary benchmark for [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter), developed by Shanghai AI Laboratory, UCLA, and CUHK.

- **[2023.03.30]** The ScienceQA dataset is now included at [OpenDataLab](https://opendatalab.org.cn/ScienceQA).

- **[2023.04.01]** Our ScienceQA dataset was downloaded **377** times in March at [HuggingFace Datasets](https://huggingface.co/datasets/derek-thomas/ScienceQA).

- **[2023.04.01]** Our work is covered by [Towards AI](https://towardsai.net/p/machine-learning/chain-of-thought-reasoning-by-c-jarnach-medium?amp=1).

- **[2023.04.01]** Our work is accepted by [CVPR 2023 O-DRUM Workshop](https://asu-apg.github.io/odrum/).

- **[2023.04.17]** [LLaVA](https://llava-vl.github.io/): A collaborative effort by UW–Madison and Microsoft, this groundbreaking work sets a new SOTA at **92.53%**. :star:

- **[2023.04.19] **[Chameleon](https://chameleon-llm.github.io/): Developed by UCLA and Microsoft, this innovative project achieves a new SOTA in the few-shot setting, reaching an impressive **86.54%**. :star:

- **[2023.04.30]** In April, our ScienceQA dataset was downloaded **1,255** times from [HuggingFace Datasets](https://huggingface.co/datasets/derek-thomas/ScienceQA), showcasing its growing popularity in the community. [[Link](https://raw.githubusercontent.com/lupantech/ScienceQA/main/assets/huggingface_2023.04.30.png)]

  

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lupantech/ScienceQA&type=Date)](https://star-history.com/#lupantech/ScienceQA&Date)



## :fire: Leaderboard :fire:

:bell: The leaderboard is continuously being updated. If you have any new results to contribute, please feel free to reach out to us.

| **#** | **Method**             |                         **Sources**                          | **Date** |    #Size    |   #Param    |   **Avg**    |   **NAT**    |   **SOC**    |   **LAN**    |   **TXT**    |   **IMG**    |       **NO** | **G1-6**     | **G7-12**    |
| :---: | ---------------------- | :----------------------------------------------------------: | :------: | :---------: | :---------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | -----------: | ------------ | ------------ |
|   1   | **Random Chance**      |       [NeurIPS 2022](https://arxiv.org/abs/2209.09513)       | 09/2022  |      -      |      -      |    39.83     |    40.28     |    46.13     |    29.25     |    47.45     |    40.08     |        33.66 | 39.35        | 40.67        |
|   2   | **Human Average**      |       [NeurIPS 2022](https://arxiv.org/abs/2209.09513)       | 09/2022  |      -      |      -      | <u>88.40</u> | <u>90.23</u> | <u>84.97</u> | <u>87.48</u> | <u>89.60</u> | <u>87.50</u> | <u>88.10</u> | <u>91.59</u> | <u>82.42</u> |
|   3   | **MCAN**               | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.pdf) | 09/2022  |     95M     |     95M     |    54.54     |    56.08     |    46.23     |    58.09     |    59.43     |    51.17     |        55.40 | 51.65        | 59.72        |
|   4   | **Top-Down**           |        [CVPR 2018](https://arxiv.org/abs/1707.07998)         | 09/2022  |     70M     |     70M     |    59.02     |    59.50     |    54.33     |    61.82     |    62.90     |    54.88     |        59.79 | 57.27        | 62.16        |
|   5   | **BAN**                |       [NeurIPS 2018](https://arxiv.org/abs/1805.07932)       | 09/2022  |    112M     |    112M     |    59.37     |    60.88     |    46.57     |    66.64     |    62.61     |    52.60     |        65.51 | 56.83        | 63.94        |
|   6   | **DFAF**               | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Dynamic_Fusion_With_Intra-_and_Inter-Modality_Attention_Flow_for_Visual_CVPR_2019_paper.pdf) | 09/2022  |     74M     |     74M     |    60.72     |    64.03     |    48.82     |    63.55     |    65.88     |    54.49     |        64.11 | 57.12        | 67.17        |
|   7   | **ViLT**               |        [ICML 2021](https://arxiv.org/abs/2102.03334)         | 09/2022  |    113M     |    113M     |    61.14     |    60.48     |    63.89     |    60.27     |    63.20     |    61.38     |        57.00 | 60.72        | 61.90        |
|   8   | **Patch-TRM**          |       [NeurIPS 2021](https://arxiv.org/abs/2110.13214)       | 09/2022  |     90M     |     90M     |    61.42     |    65.19     |    46.79     |    65.55     |    66.96     |    55.28     |        64.95 | 58.04        | 67.50        |
|   9   | **VisualBERT**         |   [ACL 2020](https://aclanthology.org/2020.acl-main.469/)    | 09/2022  |    111M     |    111M     |    61.87     |    59.33     |    69.18     |    61.18     |    62.71     |    62.17     |        58.54 | 62.96        | 59.92        |
|  10   | **UnifiedQA**          |        [EMNLP 2020](https://arxiv.org/abs/2005.00700)        | 09/2022  |    223M     |    223M     |    70.12     |    68.16     |    69.18     |    74.91     |    63.78     |    61.38     |        77.84 | 72.98        | 65.00        |
|  11   | **UnifiedQA (CoT)**    |       [NeurIPS 2022](https://arxiv.org/abs/2209.09513)       | 09/2022  |    223M     |    223M     |    74.11     |    71.00     |    76.04     |    78.91     |    66.42     |    66.53     |        81.81 | 77.06        | 68.82        |
|  12   | **GPT-3** (2-shot)     |       [NeurIPS 2020](https://arxiv.org/abs/2005.14165)       | 09/2022  |    173B     |     0M      |    73.97     |    74.64     |    69.74     |    76.00     |    74.44     |    67.28     |        77.42 | 76.80        | 68.89        |
|  13   | **GPT-3** (zero-shot)  |       [NeurIPS 2020](https://arxiv.org/abs/2005.14165)       | 09/2022  |    173B     |     0M      |    74.04     |    75.04     |    66.59     |    78.00     |    74.24     |    65.74     |        79.58 | 76.36        | 69.87        |
|  14   | **GPT-3.5 (CoT)** (AE) |       [NeurIPS 2022](https://arxiv.org/abs/2209.09513)       | 09/2022  |    173B     |     0M      |    74.61     |    76.60     |    65.92     |    77.55     |    75.51     |    66.09     |        79.58 | 78.49        | 67.63        |
|  15   | **GPT-3 (CoT)** (ALE)  |       [NeurIPS 2022](https://arxiv.org/abs/2209.09513)       | 09/2022  |    173B     |     0M      |    75.17     |    75.44     |    70.87     |    78.09     |    74.68     |    67.43     |        79.93 | 78.23        | 69.68        |
|  16   | **Multimodal-CoT** (T) |     [arXiv 2302.00923](https://arxiv.org/abs/2302.00923)     | 02/2023  | <u>223M</u> |    223M     |    70.53     |    71.09     |    70.75     |    69.18     |    71.16     |    65.84     |        71.57 | 71.00        | 69.68        |
|  17   | **Multimodal-CoT**     |     [arXiv 2302.00923](https://arxiv.org/abs/2302.00923)     | 02/2023  | <u>223M</u> |    223M     |  84.91   |  87.52   |  77.17   |  85.82   |  87.88   |  82.90   |    86.83 | 84.65    | 85.37    |
|  18   | **LLaMA-Adapter** (T)  |     [arXiv 2303.16199](https://arxiv.org/abs/2303.16199)     | 03/2023  |     6B      | <u>1.2M</u> |    78.31     |    79.00     |    73.79     |    80.55     |    78.30     |    70.35     |        83.14 | 79.77        | 75.68        |
| 19| **LLaMA-Adapter** | [arXiv 2303.16199](https://arxiv.org/abs/2303.16199) | 03/2023 | 6B | <u>1.2M</u> | **85.19** | **84.37** | **88.30** | **84.36** | **83.72** | **80.32** | **86.90** | **85.83** | **84.05** |
|  20   | **LLaVa**    |     [arXiv 2304.08485](https://arxiv.org/abs/2304.08485)     | 04/2023  |     13B     | 13B |    90.92     |    90.36     |    95.95     |    88.00     |    89.49     |    88.00     |        90.66 | 90.93        | 90.90        |
|  21   | **LLaVa (GPT-4)**    |     [arXiv 2304.08485](https://arxiv.org/abs/2304.08485)     | 04/2023  |     13B     | 13B |    **92.53** | **91.56** | **96.74** | **91.09** | **90.62** | **88.99** | **93.52** | **92.73** | **92.16** |
|  22   | **ChatGPT CoT** |     [arXiv 2304.09842](https://arxiv.org/abs/2304.09842)     | 04/2023  |     -      | 0M | 78.31 | 78.82 | 70.98 | 83.18 | 77.37 | 67.92 | 86.13 | 80.72 | 74.03 |
|  23   | **GPT-4 CoT**  |     [arXiv 2304.09842](https://arxiv.org/abs/2304.09842)     | 04/2023  |     -      | 0M | 83.99 | 85.48 | 72.44 | 90.27 | 82.65 | 71.49 | 92.89 | 86.66 | 79.04 |
|  24   | **Chameleon (ChatGPT)**  |     [arXiv 2304.09842](https://arxiv.org/abs/2304.09842)     | 04/2023  |     -     | 0M | 79.93 | 81.62 | 70.64 | 84.00 | 79.77 | 70.80 | 86.62 | 81.86 | 76.53 |
|  25   | **Chameleon (GPT-4)**  |     [arXiv 2304.09842](https://arxiv.org/abs/2304.09842)     | 04/2023  |     -      | 0M | **86.54** | **89.83** | **74.13** | **89.82** | **88.27** | **77.64** | **92.13** | **88.03** | **83.72** |

Some notations in the table

- #Size: the model size (number of model parameters)
- #Param: number of tuned model parameters
- GPT-3: the `text-davinci-002` engine
- GPT-4: the `gpt-4` engine
- AE: the output is the answer followed by the explanation
- ALE: the output is the answer followed by the lecture and explanation
- T: the input only involves textual features
- **Natural, Social, and Language Sciences**: Questions are categorized into three subject areas: NAT (Natural Science), SOC (Social Science), and LAN (Language Science)
- **Context Types**: Questions are organized based on the type of context they include: TXT (Textual), IMG (Image), and NO (No Context)
- **Grade Levels**: Questions are divided into two groups based on the targeted grade levels: G1-6 (Grades 1-6) and G7-12 (Grades 7-12)



## :world_map: About ScienceQA

We present **Science Question Answering (ScienceQA)**, a new benchmark that consists of 21,208 multimodal multiple choice questions with a diverse set of *science* topics and annotations of their answers with corresponding *lectures* and *explanations*. The lecture and explanation provide general external knowledge and specific reasons, respectively, for arriving at the correct answer.

![scienceqa](assets/scienceqa.png)

ScienceQA, in contrast to previous datasets, has richer domain diversity from three subjects: **natural science**, **language science**, and **social science**. ScienceQA features 26 topics, 127 categories, and 379 skills that cover a wide range of domains.

![domains](assets/domains.png)

We further design language models to learn to generate lectures and explanations as **the chain of thought (CoT)** to mimic the multi-hop reasoning process when answering ScienceQA questions. ScienceQA demonstrates the utility of CoT in language models, as CoT improves the question answering performance by 1.20% in few-shot GPT-3 and 3.99% in fine-tuned UnifiedQA.

For more details, you can find our project page [here](https://scienceqa.github.io/) and our paper [here](https://lupantech.github.io/papers/neurips22_scienceqa.pdf).



## :ghost: Download the Dataset

The text part of the **ScienceQA** dataset is provided in [data/scienceqa/problems.json](https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json). You can download the image data of ScienceQA by running:

```sh
. tools/download.sh
```

Alternatively, you can download **ScienceQA** from [Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev?usp=sharing) and unzip the images under `root_dir/data`.

💥 The **ScienceQA** dataset is now available at [HuggingFace Datasets](https://huggingface.co/datasets/derek-thomas/ScienceQA)!



## :smiling_imp: Explore ScienceQA

For more details, you can explore the datatset and check the visualizations here: [Explore](https://scienceqa.github.io/explore.html) and [Visualizations](https://scienceqa.github.io/visualize.html).

![explore](assets/explore.png)



## :octopus: Requirements

```
python==3.8.10
huggingface-hub
nltk==3.5
numpy==1.23.2
openai==0.23.0
pandas==1.4.3
rouge==1.0.1
sentence-transformers==2.2.2
torch==1.12.1+cu113
transformers==4.21.1
```

Install all required python dependencies:

```
pip install -r requirements.txt
```



## :robot: Run the GPT-3 (CoT) Model for ScienceQA

### Generate the image captions

We use the image captioning model to generate the text content for images in ScienceQA. The pre-generated image captions are provided in [data/captions.json](https://github.com/lupantech/ScienceQA/blob/main/data/captions.json).

(Optionally) You can generate the image captions with user-specific arguments with the following command, which will save the caption data in `data/captions_user.json`.

```sh
cd tools
python generate_caption.py
```

### Run the model

We build a few-shot GPT-3 model via chain-of-thought (CoT) prompting to generate the answer followed by the lecture and the explanation (**QCM→ALE**). The prompt instruction encoding for the test example in GPT-3 (CoT) is defined as below:

![scienceqa](assets/prompt.png)

In our final model, we develop GPT-3 (CoT) prompted with two in-context examples and evalute it on the ScienceQA test split:

```sh
cd models
python run_gpt3.py \
--label exp1 \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format QCM-ALE \
--seed 3
```

### Evaluate the results

Our final GPT-3 (CoT) model achieves a state-of-the-art accuracy of 75.17% on the test split. One prediction example is visualized below. We can see that GPT-3 (CoT) predicts the correct answer and generates a reasonable lecture and explanation to mimic the human thought process.

![scienceqa](assets/prediction.png)

We can get the accuracy metrics on average and across different question classes by running:

```sh
cd tools
python evaluate_acc.py
```

We can run the following command to evaluate the generated lectures and explanations automatically:

```sh
cd tools
python evaluate_explaination.py
```

### Try different prompt templates

You can try other prompt templates. For example, if you want the model to take the question, the context, and the multiple options as input, and output the answer after the lecture and explanation (QCM→LEA), you can run the following script:

```sh
cd models
python run_gpt3.py \
--label exp1 \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format QCM-LEA \
--seed 3
```



## :book: Related Work

- :fire: **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering,** NeurIPS 2022 [[ArXiv](https://arxiv.org/abs/2209.09513)] [[Project](https://scienceqa.github.io/)] [[Code](https://github.com/lupantech/ScienceQA)] [[Hugging Face Dataset](https://huggingface.co/datasets/derek-thomas/ScienceQA)]
- :fire: **Multimodal Chain-of-Thought Reasoning in Language Models**, arXiv:2302.00923 [[ArXiv](https://arxiv.org/abs/2302.00923)] [[Code](https://github.com/amazon-science/mm-cot)]
- :fire: **Visual Instruction Tuning**, arXiv:2304.08485 [[ArXiv](https://arxiv.org/abs/2304.08485)] [[Project](https://llava-vl.github.io/)] [[Code](https://github.com/haotian-liu/LLaVA)] [[Demo](https://llava.hliu.cc/)] [[Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)]
- :fire: **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**, arXiv:2304.09842 [[ArXiv](https://arxiv.org/abs/2304.09842)] [[Project](https://chameleon-llm.github.io/)] [[Code](https://github.com/lupantech/chameleon-llm)] [[YouTube](https://www.youtube.com/watch?v=EWFixIk4vjs&ab_channel=WorldofAI)]



## :warning: Licenses

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This work is licensed under a [MIT License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-SA 4.0](https://camo.githubusercontent.com/bdc6a3b8963aa99ff57dfd6e1e4b937bd2e752bcb1f1936f90368e5c3a38f670/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d434325323042592d2d5341253230342e302d6c69676874677265792e737667)](https://creativecommons.org/licenses/by-sa/4.0/)

The ScienceQA dataset is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).



## :white_check_mark: Cite

**If the paper, codes, or the dataset inspire you, please kindly cite us:**

```latex
@inproceedings{lu2022learn,
    title={Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
    author={Lu, Pan and Mishra, Swaroop and Xia, Tony and Qiu, Liang and Chang, Kai-Wei and Zhu, Song-Chun and Tafjord, Oyvind and Clark, Peter and Ashwin Kalyan},
    booktitle={The 36th Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```
