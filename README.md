# MemoryBench

**MemoryBench** aims to provide a standardized and extensible benchmark for evaluating memory and continual learning in LLM systems — encouraging future work toward more adaptive, feedback-driven, and efficient LLM systems.

- 📢 **Dec. 5, 2025 Updated**: We released a new version of user feedback data where `Mistral-Small-3.2-24B-Instruct-2506` acts as the User Simulator!

## 🌟 Introduction

Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. 

Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. 
Experiments show that the effectiveness and efficiency of state-ofthe-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.

> This repository provides a lightweight interface for **loading the MemoryBench dataset**,  **evaluations** and **follow our experiments**. For full experiments in the paper, please refer to [https://github.com/LittleDinoC/MemoryBench-code](https://github.com/LittleDinoC/MemoryBench-code).

## 🏡 Repository Structure

Below is the directory structure of MemoryBench.
Please maintain this structure to ensure correct dataset execution.
The rest of the repository contains implementations of memory systems and experiment scripts.


```plain
configs/
    datasets/           # Dataset configuration files
    final_evaluate_summary_wo_details.json # Normalization data
raw/                    # Raw datasets
src/
    datasets/           # Dataset classes
    agents/             # Evaluation agents and memory system baselines
    *.py                # Main script for some experiments
memorybench.py          # Main entry of MemoryBench
.env                    # Environment variables for evaluation configuration
```

Dataset configurations are located in `configs/datasets/`:

* `each.json` — metadata for each dataset
* `domain.json` — datasets grouped by domain
* `task.json` — datasets grouped by task

We also provide lightweight dataset loading and evaluation functions in `memorybench.py`.

## 🚀 Using MemoryBench

This part shows how to load the MemoryBench dataset and perform evaluation. If you would like to conduct experiments in our paper with your memory systems, please refer to the next section [🎯 Following Our Experiments](https://github.com/LittleDinoC/MemoryBench?tab=readme-ov-file#-following-our-experiments).

### Environment Setup

Use the following commands to set up the conda environment:

```
conda create -n memorybench python=3.10
conda activate memorybench
pip install -r requirements.txt
```

Please set up the `.env` file to specify evaluation models and OpenAI API configurations.
If you have cloned our dataset locally, you can modify `MEMORY_BENCH_PATH` to point to your local folder to use the local data.
We recommend that you use vLLM to deploy the official evaluation model [WritingBench-Critic-Model-Qwen-7B](https://huggingface.co/AQuarterMile/WritingBench-Critic-Model-Qwen-7B) and set the vLLM url in `.env`.
The evaluation models you set are used for all other LLM-as-judge evaluations and integrated scoring across multiple metrics. 

### Load Dataset

You can load datasets using the `load_memory_bench` function in `memorybench.py`.

**Parameters:**

* `dataset_type` (`single` | `domain` | `task`):
  Choose to load a single dataset, or merge datasets by domain or by task.
* `name` (str):
  The name of the dataset/domain/task.

  * Datasets are listed on the Hugging Face page.
  * Domains include `Open-Domain`, `Academic&Knowledge`, and `Legal` (see `configs/datasets/domain.json`).
  * Tasks include `Long-Short`, `Long-Long`, `Short-Long`, and `Short-Short` (see `configs/datasets/task.json`).
* `eval_mode` (bool, default=False):
  Whether to enable evaluation mode.

If `dataset_type` is `single`, the function returns a dataset instance; if `domain` or `task`, it returns a list of dataset instances.

The returned dataset class contains the following attributes and methods:

+ `dataset_name` (str): Name of the dataset.
+ `dataset`: HuggingFace dataset object, containing "train" and "test" splits.
+ `has_corpus` (bool): Whether the dataset includes a corpus, such as LoCoMo and DialSim. If true, it also contains `corpus` (list[dict]) and `session_cnt` (int) representing the dialogue corpus and number of sessions.
+ `get_data(test_idx: int) -> data`: Get the data point for the specified index, regardless of whether the data is in training set or test set.

**Example usage:**

```python
from memorybench import load_memory_bench

# Load a single dataset (JRE-L)
dataset = load_memory_bench(dataset_type='single', name='JRE-L')
print(dataset.dataset_name)  # Output: JRE-L
print(dataset.dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['test_idx', 'input_prompt', 'dataset_name', 'lang', 'info', 'dialog', 'implicit_feedback'],
        num_rows: 200
    })
    test: Dataset({
        features: ['test_idx', 'input_prompt', 'dataset_name', 'lang', 'info', 'dialog', 'implicit_feedback'],
        num_rows: 50
    })
})
"""

# Load a domain (Open-Domain) without dataset instances
dataset_list = load_memory_bench(dataset_type='domain', name='Open-Domain')

# Load a task (Long-Short, LiSo) with evaluation mode
dataset_list = load_memory_bench(dataset_type='task', name='Long-Short', eval_mode=True)
```

### Evaluation

Before starting the evaluation, make sure you have correctly configured the evaluation models in the .env file:

1. WritingBench Dataset 

    For evaluating on the WritingBench dataset, please deploy the official evaluation model [WritingBench-Critic-Model-Qwen-7B](https://huggingface.co/AQuarterMile/WritingBench-Critic-Model-Qwen-7B). You can launch it with vLLM using the following command:

    ```python
    vllm serve AQuarterMile/WritingBench-Critic-Model-Qwen-7B --port 12388 # LLM as Judge (for WritingBench)
    ```

2. Other Datasets using LLM-as-judge (e.g., NFCats, IdeaBench, etc.)

    For datasets that rely on black-box evaluation models, configure the corresponding API key in your `.env` file. All datasets requiring black-box evaluators will use this configuration.
    
    In our paper, we use DeepSeek-V3 as the evaluation model for these benchmarks.

Once the environment is properly set up, you can run evaluation using the `evaluate` function.

**❗Note:** Some datasets such as `JRE-L` use `bert_score` library to evaluate. And there is a bug in this lib: model loaded locally can't automatically truncate the inputs (see [Issues](https://github.com/Tiiiger/bert_score/issues?q=truncate)). So please load the model from huggingface as we do in this repo, instead of loading locally, or you will meet some "exceeding max length" errors during evaluating.

**Parameters:**

* `dataset_type` (`single` | `domain` | `task`): same as above.
* `name` (str): dataset/domain/task name.
* `predicts` (list of dict): list of model predictions.
  Each element must include:

  * `dataset` (str): dataset name.
  * `test_idx` (int): index of the test sample.
  * `response` (str): model’s response.

The function loads the datasets, runs the evaluation, and returns a list of results:


```python
{
    "dataset": str,      # Dataset name
    "test_idx": int,     # Sample index
    "metrics": {         # Evaluation metrics
        "metric_name_1": value_1,
    }
}
```


**Example usage:**

```python
from memorybench import evaluate

evaluate_details = evaluate(
    dataset_type='domain',
    name='Open-Domain',
    predicts=[
        {"test_idx": 0, "response": "Hello World!", "dataset": "WritingPrompts"},
        {"test_idx": 1, "response": "Another response.", "dataset": "DialSim-friends"},
    ]
)
```


### Summary and Normalization

The `evaluate` function produces per-sample metrics.
To compute overall performance scores, please use the `summary_results` function.

For single datasets, it computes the mean of each metric directly.
For domains or tasks, it additionally performs normalization across datasets using precomputed statistics.


**Parameters:**

* `dataset_type` (`single` | `domain` | `task`): same as above.
* `name` (str): dataset/domain/task name.
* `predicts` (list of dict): model predictions.
* `evaluate_details` (list of dict): detailed results from the `evaluate` function.
* `min_max_config_file` (str, default=`configs/final_evaluate_summary_wo_details.json`):
  Configuration file containing normalization parameters (min, max, mean, std).

The function returns a dictionary whose core field is `summary`, containing the averaged or normalized results.

**Example usage:**

```python
from memorybench import evaluate, summary_results

predicts = [
    {"test_idx": 0, "response": "Your model's response here.", "dataset": "WritingPrompts"},
    {"test_idx": 1, "response": "Another response.", dataset: "DialSim-friends"},
    # Add more predictions as needed
]

evaluate_details = evaluate(
    dataset_type='domain',
    name='Open-Domain',
    predicts=predicts
)

summary = summary_results(
    dataset_type='domain',
    name='Open-Domain',
    predicts=predicts,
    evaluate_details=evaluate_details
)
```

## 🎯 Following Our Experiments

Next, we will introduce how you can use our dataset to evaluate the capabilities of LLM systems . This includes how to replicate our experimental process, test your own systems, and compare the results with the baselines we provide. The section consists of the following five parts:

1. Preparation: Dialogues Generation

2. Off-policy Experiments

3. Stepwise Off-policy Experiments

4. On-policy Experiments

5. Training Performance

For the full experiments described in our paper, please refer to our [reproduction code](https://github.com/LittleDinoC/MemoryBench-code).

<details>
<summary>If you would like to directly use our code to test the baselines, click to see what you need to prepare.</summary>

Our experiments use vLLM to deploy LLM services. You need to deploy models in a similar way:

```
vllm serve Qwen/Qwen3-32B --port 12345 --chat-template qwen3_nonthinking.jinja     # Qwen3-32B
vllm serve Qwen/Qwen3-8B --port 12366 --chat-template qwen3_nonthinking.jinja      # Qwen3-8B
vllm serve Qwen/Qwen3-Embedding-0.6B --port 12377 --task embed                     # Embedding Model
vllm serve AQuarterMile/WritingBench-Critic-Model-Qwen-7B --port 12388             # Evaluation Model for WritingBench
```

If you deploy using these commands, you don't need to modify configuration files in `configs/memory_systems`. Otherwise, adjust the configuration files based on your own setup. Model configuration details can be found in `src/llms/`.

We use Qwen3’s non-thinking mode via the official vLLM configuration. You can find more details in the [documents](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes).

For each memory system, the correspondence between paper names, code names, and configuration files (in the `configs/memory_systems`) is shown below:

| Paper Name |	Code Name	| Config File |
|--------------|----------------|----------------|
| Vanilla  | wo_memory           | base.json |
| BM25-M | bm25_message       | bm25.json |
| BM25-S | bm25_dialog | bm25.json |
| Emb-M | embedder_message  | embedder.json |
| Emb-S | embedder_dialog   | embedder.json |
| A-Mem | a_mem | a_mem.json |
| Mem0 | mem0 | mem0.json |
| MemoryOS | memoryos | memoryos.json |

You can modify these configuration files to adjust parameters of each memory system.
Their implementations can be found under `src/agent/`.

</details>

If you want to use the main experimental scripts, you can also implement your own memory system by following the interfaces provided in `src/agent/` and `src/solver/`.


### 1. Preparation: Dialogues Generation

For general datasets, we provide dialogues between the vanilla LLM and the User Feedback Simulator, which can be found in the `dialog` field of each data point.

For datasets with corpora, such as LoCoMo and DialSim, since different systems has different memory behaviors, we seperately generate dialogues for each system. We provide dialogues generated before, which can be found in the `dialog_xxx` fields of each data point. For new systems, you can follow the behavior of the main script `src/generate_dialogs/reading.py` to generate dialogues, mainly including:

1. Store the corpus into the system’s memory.

2. For all data points in the training set, conduct conversations with the User Feedback Simulator:

+ In the first round, use the original question and the retrieved memories as input.

+ In subsequent rounds, use only the user feedback without retrieving additional memories.

<details>
<summary>Click to view the instructions for the dialogue generation script.</summary>

You can run `run_scripts/create_dialogs.py` to generate dialogues for each dataset.  

For datasets without a corpus, the main script is `src/generate_dialogs/basic.py`, which by default uses configurations `base.json` and `feedback.json` in the `configs/memory_systems/` directory for the vanilla LLM agent and the feedback agent, respectively. You can specify the dataset name with `--dataset`.

Example command:

```bash
python -m src.generate_dialogs.basic --dataset JRE-L
```

For LoCoMo and DialSim, the main script is `src/generate_dialogs/reading.py`.
You need to specify the memory system using `--memory_system` (matching a config file in `configs/memory_systems/`) and the dataset name using `--dataset`.

Example command:

```bash
python -m src.generate_dialogs.reading --memory_system bm25_message --dataset Locomo-0
```

</details>

### 2. Off-policy Experiments

For all baselines, you can run the off-policy experiments using `python run_scripts/off_policy.py`.

For new systems, you can refer to our main script `src/off-policy.py` to implement the off-policy experiments. The process mainly consists of the following steps:

1. Load all training dialogues from the datasets, mix and shuffle them, and store them in the system memory.

2. For the test data in each dataset, retrieve relevant memories to answer the questions.
For datasets containing a corpus, the memory will include both the training dialogues from all datasets and the corpus of the current dataset when answering questions from that dataset.

<details>

<summary>Click to view usage instructions of the main script.</summary>

The main script for running all off-policy experiments is `src/predict.py`. You need to specify the memory system using `--memory_system`, the `--dataset_type` (`domain` or `task`), and the `--set_name`(the specific domain/task name). 

Example command:

```bash
python -m src.off-policy --memory_system bm25_message --dataset_type domain --set_name Open-Domain
```

You can find more detailed parameter configuration in the code, including the number of single retrievals (`--retrieve_k`, default is 5), etc. The results are stored in `off-policy/results` by default.

Since the Mem0 method takes too long to run on Open-Domain and LiSo tasks,
`run_scripts/off_policy.py` skips it by default — you can enable it manually if needed.

</details>

### 3. Stepwise Off-policy Experiments

For all baselines, you can run the stepwise off-policy experiments using `python run_scripts/stepwise_off-policy.py`.

For new systems, you can refer to our main script `src/stepwise_off-policy.py` to implement the stepwise off-policy experiments. The process mainly consists of the following steps:

1. If the dataset contains a corpus, load the corpus into memory first.

2. Load all training dialogues, shuffle them, and split them into batches.

3. For each step:

    + Load the next batch of training dialogues into memory.

    + Test the system on the test data after the new memory has been added.

<details> 

<summary>Click to view usage instructions of the main script.</summary>

The main script is `src/stepwise_off-policy.py`, sharing most of configuration options with the off-policy setup.
Specifically, you can specify the `--batch_size` of dialogues to be memorized in a single step, which defaults to 100.
The results are stored in `step_off-policy/results` by default.

Example command:

```bash
python -m src.stepwise_off-policy --memory_system bm25_message --dataset_type domain --set_name Open-Domain
```

</details>

### 4. On-policy Experiments

For all baselines, you can run the on-policy experiments using `python run_scripts/on-policy.py`.

For new systems, you can refer to our main script `src/on-policy.py` to implement the on-policy experiments. The process mainly consists of the following steps:

1. If the dataset contains a corpus, load the corpus into memory first.

2. Load all training dialogues, shuffle them.

3. For each step:

    + Sample a batch of training data points randomly and conduct conversations with the User Feedback Simulator to generate new dialogues (similar to the dialogue generation process described above).

    + Load the generated dialogues into memory.

    + Test the system on the test data after the new memory has been added.

<details>

<summary>Click to view usage instructions of the main script.</summary>

The main script is `src/on_policy.py`, sharing most of configuration options with the off-policy setup.
You can set `--max_rounds` to specify the number of conversation rounds (default is 3), `--batch_size` to specify the number of conversations to remember at a time (default is 100), and `--step` to specify the number of runs (default is 10)
The results are stored in `on-policy/results` by default.

Example command:

```bash
python -m src.stepwise_off-policy --memory_system bm25_message --domain Open-Domain --dataset_config configs/datasets/domain.json
```

</details>

### 5. Training Performance

For all baselines, you can run the off-policy experiments on training sets using `python run_scripts/train_performance.py`.

For new systems, you can refer to our main script `src/train_performance.py` to implement the off-policy experiments. The process mainly consists of the following steps:

1. Load all training dialogues from the datasets, mix and shuffle them, and store them in the system memory.

2. For the **training** data in each dataset, retrieve relevant memories to answer the questions.
For datasets containing a corpus, the memory will include both the training dialogues from all datasets and the corpus of the current dataset when answering questions from that dataset.

<details>

<summary>Click to view usage instructions of the main script.</summary>

This experiment shares all configurations with off-policy experiments,
using the same dialogues — the only difference is that answers are generated for the questions in the training set.

The main script is `src/train_performance.py` and the results are stored in `train_performance/results` by default.

Example command:

```bash
python -m src.train_performance --memory_system bm25_message --dataset_type domain --set_name Open-Domain
```
