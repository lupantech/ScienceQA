# ScienceQA: Science Question Answering

![VQA](https://img.shields.io/badge/Task-VQA-orange) ![Science Problems](https://img.shields.io/badge/Task-Science_Problems-orange) ![ScienceQA](https://img.shields.io/badge/Dataset-ScienceQA-blue) ![Chain-of-Thought](https://img.shields.io/badge/Model-Chain_of_Thought-green) ![GPT-3](https://img.shields.io/badge/Model-GPT--3-green)  ![LLMs](https://img.shields.io/badge/Model-LLMs-green)

Data and code for NeurIPS 2022 Paper "[Learn to Explain: Multimodal Reasoning via
Thought Chains for Science Question Answering](http://lupantech.github.io/papers/neurips22_scienceqa.pdf)".

For more details, please refer to the project page with dataset exploration and visualization tools: https://scienceqa.github.io.

:bell: If you have any questions or suggestions, please don't hesitate to let us know. You can directly email [Pan Lu](https://lupantech.github.io/) at UCLA using the email address lupantech@gmail.com, comment on the [Twitter](https://twitter.com/lupantech/status/1570828580346802178), or post an issue on this repository.



## 💥 News 💥

- [09/2022] Our work is accepted to [NeurIPS 2022](https://nips.cc/Conferences/2022).
- [11/2022] Our work is featured on [Medium](https://medium.com/geekculture/neurips-2022-the-first-multi-modal-science-question-answering-science-qa-dataset-with-detailed-45b8d628b301).
- [12/2022] Our work gives an poster presentation by Pan Lu at [NeurIPS 2022](https://nips.cc/Conferences/2022).
- [02/2023] The ScienceQA dataset has served as the primary benchmark for the new generation of multimodal reasoning systems, [Multimodal-CoT](https://github.com/amazon-science/mm-cot), developed by [Amazon Science](https://www.amazon.science/).
- [02/2023] The ScienceQA dataset is now included in [HuggingFace Datasets](https://huggingface.co/datasets/derek-thomas/ScienceQA).
- [02/2023] Our work gives an oral presentation by Pan Lu at [AAAI 2023 KnowledgeNLP Workshop](https://knowledge-nlp.github.io/aaai2023/publications.html).




## :fire: Leaderboard :fire:

:bell: The leaderboard is continuously being updated. If you have any new results to contribute, please feel free to reach out to us.

| **#** | **Method**                        | **Sources**                                                  | **Date** | **NAT** | **SOC** | **LAN** | **TXT** | **IMG** | **NO** | **G1-6** | **G7-12** | **Avg** |
| -------- | --------------------------------- | ------------------------------------------------------------ | -------- | ------- | ------- | ------- | ------- | ------- | ------ | -------- | --------- | ------- |
| 1        | Random Chance                    | [Pan et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) | Sep-2022 | 40.28   | 46.13   | 29.25   | 47.45   | 40.08   | 33.66  | 39.35    | 40.67     | 39.83   |
| 2        | Human Average                     | [Pan et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) | Sep-2022 | **90.23** | **84.97** | **87.48** | **89.60** | **87.50** | **88.10** | **91.59** | **82.42** | **88.40** |
| 3        | **GPT-3.5** (QCM→A, 2-shot)   | [Pan et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) | Sep-2022 | 74.64   | 69.74   | 76.00   | 74.44   | 67.28   | 77.42  | 76.80    | 68.89     | 73.97   |
| 4        | **GPT-3.5** (QCM→A, zero-shot)      | [Pan et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) | Sep-2022 | 75.04   | 66.59   | 78.00   | 74.24   | 65.74   | 79.58  | 76.36    | 69.87     | 74.04   |
| 5        | **GPT-3.5 w/ CoT** (QCM→A, 2-shot) | [Pan et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) | Sep-2022 | 76.60   | 65.92   | 77.55   | 75.51   | 66.09   | 79.58  | 78.49    | 67.63     | 74.61   |
| 6        | **GPT-3.5 w/ CoT** (QCM→ALE, 2-shot) | [Pan et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) | Sep-2022 | 75.44   | 70.87   | 78.09   | 74.68   | 67.43   | 79.93  | 78.23    | 69.68     | 75.17   |
| 7        | **Multimodal-CoT** (Base) | [Zhang et al., arXiv 2023](https://arxiv.org/abs/2302.00923) | Feb-2023 | 87.52     | 77.17     | 85.82     | 87.88     | 82.90     | 86.83     | 84.65     | 85.37     | 84.91     |
| 8        | **Multimodal-CoT** (Large) | [Zhang et al., arXiv 2023](https://arxiv.org/abs/2302.00923) | Feb-2023 | **95.91** | **82.00** | **90.82** | **95.26** | **88.80** | **92.89** | **92.44** | **90.31** | **91.68** |



## About ScienceQA

We present **Science Question Answering (ScienceQA)**, a new benchmark that consists of 21,208 multimodal multiple choice questions with a diverse set of *science* topics and annotations of their answers with corresponding *lectures* and *explanations*. The lecture and explanation provide general external knowledge and specific reasons, respectively, for arriving at the correct answer.

![scienceqa](data/scienceqa.png)

ScienceQA, in contrast to previous datasets, has richer domain diversity from three subjects: **natural science**, **language science**, and **social science**. ScienceQA features 26 topics, 127 categories, and 379 skills that cover a wide range of domains.

![domains](data/domains.png)

We further design language models to learn to generate lectures and explanations as **the chain of thought (CoT)** to mimic the multi-hop reasoning process when answering ScienceQA questions. ScienceQA demonstrates the utility of CoT in language models, as CoT improves the question answering performance by 1.20% in few-shot GPT-3 and 3.99% in fine-tuned UnifiedQA.

For more details, you can find our project page [here](https://scienceqa.github.io/) and our paper [here](https://lupantech.github.io/papers/neurips22_scienceqa.pdf).



## Download the dataset

The text part of the **ScienceQA** dataset is provided in [data/scienceqa/problems.json](https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json). You can download the image data of ScienceQA by running:

```sh
. tools/download.sh
```

Alternatively, you can download **ScienceQA** from [Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev?usp=sharing) and unzip the images under `root_dir/data`.

💥 The **ScienceQA** dataset is now available at [HuggingFace Datasets](https://huggingface.co/datasets/derek-thomas/ScienceQA)!



## Requirements

```
python==3.8.10
huggingface-hub==0.0.12
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



## Run the GPT-3 (CoT) Model for ScienceQA

### Generate the image captions

We use the image captioning model to generate the text content for images in ScienceQA. The pre-generated image captions are provided in [data/captions.json](https://github.com/lupantech/ScienceQA/blob/main/data/captions.json).

(Optionally) You can generate the image captions with user-specific arguments with the following command, which will save the caption data in `data/captions_user.json`.

```sh
cd tools
python generate_caption.py
```

### Run the model

We build a few-shot GPT-3 model via chain-of-thought (CoT) prompting to generate the answer followed by the lecture and the explanation (**QCM→ALE**). The prompt instruction encoding for the test example in GPT-3 (CoT) is defined as below:

![scienceqa](data/prompt.png)

In our final model, we develp GPT-3 (CoT) prompted with two in-context examples and evalute it on the ScienceQA test split:

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

Our final GPT-3 (CoT) model achieves a state-of-the-art accuracy of 75.17% on the test split. One prediction example is visualized bellow. We can see that GPT-3 (CoT) predicts the correct answer and generates a reasonable lecture and explanation to mimic the human thought process.

![scienceqa](data/prediction.png)

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



## License

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This work is licensed under a [MIT License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-SA 4.0](https://camo.githubusercontent.com/bdc6a3b8963aa99ff57dfd6e1e4b937bd2e752bcb1f1936f90368e5c3a38f670/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d434325323042592d2d5341253230342e302d6c69676874677265792e737667)](https://creativecommons.org/licenses/by-sa/4.0/)

The ScienceQA dataset is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).



## Cite

**If the paper, codes, or the dataset inspire you, please cite us:**

```latex
@inproceedings{lu2022learn,
    title={Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
    author={Lu, Pan and Mishra, Swaroop and Xia, Tony and Qiu, Liang and Chang, Kai-Wei and Zhu, Song-Chun and Tafjord, Oyvind and Clark, Peter and Ashwin Kalyan},
    booktitle={The 36th Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```

