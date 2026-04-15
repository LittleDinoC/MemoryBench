# MemoryBench Frontend (Streamlit)

![](./assets/1.png)

This frontend supports:

- Running `off-policy` and `on-policy` experiments.
- Configuring API-based model access (OpenAI or OpenAI-compatible endpoint).
- Choosing memory systems from the existing benchmark methods.
- Monitoring live experiment logs.
- Browsing results and every dialogue item, including the assembled memory context prompt.

> *Hint:* You can use Code Agent like Claude Code to quickly configure the environment and launch the frontend. You can prompt the agent with ***"Help me set up the environment and launch the MemoryBench frontend for me under the instruction in frontend/README.md"*** or similar instructions. 
> See [For Code Agent](./README-for-code-agent.md) for more details.


## 0. Install project dependencies

```bash
pip install -r requirements.txt
cd baselines/mem0
pip install -e .
```

Then install nltk data in python:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

Download Huggingface Dataset to local directory (optional, but can speed up the first run):

```bash
huggingface-cli download --repo-type dataset --resume-download THUIR/MemoryBench --local-dir /path/to/MemoryBench
```


## 1. Install frontend dependency

```bash
pip install -r frontend/requirements.txt
```

## 2. Launch frontend

```bash
python -m streamlit run frontend/streamlit_app.py
```

## 3. Key notes

- The app only writes new runtime artifacts:
  - `frontend/runtime_configs/` for temporary configs.
  - Your chosen output directory for experiment results.



## For Code Agent

Use these commands from the repository root to set up the environment and launch the frontend:

If `conda` is available, it's recommended to create a new environment (ask the user whether to create a new conda environment named `memorybench`). If yes, then:
```bash
conda create -n memorybench python=3.10
conda activate memorybench
```

Then:

```bash
pip install -r requirements.txt
cd baselines/mem0
pip install -e .
cd ../..
pip install -r frontend/requirements.txt
python -c "import nltk; [nltk.download(x) for x in ('punkt', 'wordnet', 'stopwords')]"
python smoke_test.py
```

`python smoke_test.py` is smoke test. You need to fix any errors until it runs successfully. If it runs successfully, then launch the frontend:

```bash
python -m streamlit run frontend/streamlit_app.py
```
