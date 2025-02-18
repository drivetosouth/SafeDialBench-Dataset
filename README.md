# SafeDialBench-Dataset
This repository contains the dataset for the paper **SafeDialBench**: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks

## Description

With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern. 
Current benchmarks primarily concentrate on single-turn dialogues or single jailbreak attack method to assess the safety of LLMs. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark (**SafeDialBench**) for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical taxonomy that considers $6$ distinct dimensions and generates more than $4000$ multi-turn dialogues in both Chinese and English under $22$ dialogue scenarios. 
We employ $7$ jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across $17$ LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities.

<img src="figs/framework.png" width="100%" height="auto" /> 

## Leaderboard

<img src="figs/leaderboard.png" width="100%" height="auto" /> 

## Cases

An example of dialogue and model evaluation for ethics under scene construct attack against GLM4-9B-Chat.

<img src="figs/case.png" width="85%" height="auto" /> 

## Cite

If you find our work helpful, feel free to cite this work.

```
@misc{cao2025safedialbenchfinegrainedsafetybenchmark,
      title={SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks}, 
      author={Hongye Cao and Yanming Wang and Sijia Jing and Ziyue Peng and Zhixin Bai and Zhe Cao and Meng Fang and Fan Feng and Boyan Wang and Jiaheng Liu and Tianpei Yang and Jing Huo and Yang Gao and Fanyu Meng and Xi Yang and Chao Deng and Junlan Feng},
      year={2025},
      eprint={2502.11090},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11090}, 
}
```
