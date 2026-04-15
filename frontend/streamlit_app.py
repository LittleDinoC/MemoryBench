import copy
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_DIR = ROOT / "frontend" / "runtime_configs"
RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CONFIG_DIR = ROOT / "configs" / "datasets"
MEMORY_CONFIG_DIR = ROOT / "configs" / "memory_systems"

OFF_POLICY_MEMORYS = [
    "wo_memory",
    "bm25_message",
    "bm25_dialog",
    "embedder_message",
    "embedder_dialog",
    "a_mem",
    "mem0",
    "memoryos",
]

ON_POLICY_MEMORYS = [
    "bm25_message",
    "bm25_dialog",
    "embedder_message",
    "embedder_dialog",
    "a_mem",
    "mem0",
    "memoryos",
]


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_choices() -> Dict[str, List[str]]:
    each = _load_json(DATASET_CONFIG_DIR / "each.json", {})
    domain = _load_json(DATASET_CONFIG_DIR / "domain.json", {})
    task = _load_json(DATASET_CONFIG_DIR / "task.json", {})
    return {
        "single": sorted(each.keys()),
        "domain": sorted(domain.keys()),
        "task": sorted(task.keys()),
    }


def load_memory_template(memory_system: str) -> Dict:
    default_map = {
        "wo_memory": "base.json",
        "a_mem": "a_mem.json",
        "bm25_message": "bm25.json",
        "bm25_dialog": "bm25.json",
        "embedder_message": "embedder.json",
        "embedder_dialog": "embedder.json",
        "mem0": "mem0.json",
        "memoryos": "memoryos.json",
    }
    filename = default_map[memory_system]
    return _load_json(MEMORY_CONFIG_DIR / filename, {})


def load_feedback_template() -> Dict:
    return _load_json(MEMORY_CONFIG_DIR / "feedback.json", {})


def build_llm_config(provider: str, model: str, base_url: str, api_key: str, temperature: float) -> Dict:
    llm_cfg = {
        "model": model,
        "temperature": temperature,
    }
    if provider == "openai":
        llm_cfg["openai_base_url"] = base_url
        if api_key:
            llm_cfg["api_key"] = api_key
    else:
        llm_cfg["vllm_base_url"] = base_url
        if api_key:
            llm_cfg["api_key"] = api_key
    return llm_cfg


def build_runtime_memory_config(
    memory_system: str,
    provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    temperature: float,
    retrieve_k: int,
    embedder_provider: str,
    embedder_model: str,
    embedder_base_url: str,
    embedder_dim: int,
) -> Dict:
    cfg = copy.deepcopy(load_memory_template(memory_system))

    if "llm_provider" in cfg:
        cfg["llm_provider"] = provider
    cfg["llm_config"] = build_llm_config(
        provider=provider,
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        temperature=temperature,
    )

    if "retrieve_k" in cfg:
        cfg["retrieve_k"] = retrieve_k

    if memory_system == "mem0":
        cfg["embedder_provider"] = embedder_provider
        cfg["embedder_config"] = {
            "model": embedder_model,
            "embedding_dims": embedder_dim,
        }
        if embedder_provider == "openai":
            cfg["embedder_config"]["base_url"] = embedder_base_url
            if llm_api_key:
                cfg["embedder_config"]["api_key"] = llm_api_key
        else:
            cfg["embedder_config"]["vllm_base_url"] = embedder_base_url
            if llm_api_key:
                cfg["embedder_config"]["api_key"] = llm_api_key

    if memory_system.startswith("embedder"):
        cfg["embedder_provider"] = embedder_provider
        cfg["embedder_model"] = embedder_model
        cfg["embedder_base_url"] = embedder_base_url
        cfg["embedding_dim"] = embedder_dim

    return cfg


def build_runtime_feedback_config(
    provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
) -> Dict:
    cfg = copy.deepcopy(load_feedback_template())
    cfg["llm_provider"] = provider
    cfg["llm_config"] = {
        "model": llm_model,
    }
    if provider == "openai":
        cfg["llm_config"]["openai_base_url"] = llm_base_url
    else:
        cfg["llm_config"]["vllm_base_url"] = llm_base_url
    if llm_api_key:
        cfg["llm_config"]["api_key"] = llm_api_key
    return cfg


def write_runtime_config(config_obj: Dict, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = RUNTIME_CONFIG_DIR / f"{prefix}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_obj, f, ensure_ascii=False, indent=2)
    return path


def build_command(
    mode: str,
    dataset_type: str,
    set_name: str,
    memory_system: str,
    memory_config_path: Path,
    output_dir: str,
    cache_prefix: str,
    threads: int,
    retrieve_k: int,
    step: int,
    batch_size: int,
    max_rounds: int,
    feedback_config_path: Optional[Path],
) -> List[str]:
    module_name = "src.off-policy" if mode == "off-policy" else "src.on-policy"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        module_name,
        "--dataset_type",
        dataset_type,
        "--set_name",
        set_name,
        "--memory_system",
        memory_system,
        "--memory_system_config",
        str(memory_config_path),
        "--output_dir",
        output_dir,
        "--memory_cache_prefix",
        cache_prefix,
        "--threads",
        str(threads),
        "--retrieve_k",
        str(retrieve_k),
    ]

    if mode == "on-policy":
        cmd.extend(
            [
                "--step",
                str(step),
                "--batch_size",
                str(batch_size),
                "--max_rounds",
                str(max_rounds),
                "--feedback_agent_config",
                str(feedback_config_path),
            ]
        )

    return cmd


def run_command_live(cmd: List[str], extra_env: Dict[str, str]) -> Tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_ASCII", "1")
    env.setdefault("TQDM_NCOLS", "80")
    env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    output_queue: "queue.Queue[str]" = queue.Queue()
    _start_output_reader(proc, output_queue)

    logs: List[str] = []
    while True:
        logs = _drain_output_lines(output_queue, logs)
        if proc.poll() is not None and output_queue.empty():
            break
        time.sleep(0.05)

    return_code = proc.wait()
    output = "\n".join(logs)
    return return_code, output


def _start_output_reader(proc: subprocess.Popen, output_queue: "queue.Queue[str]") -> None:
    def _enqueue_output() -> None:
        if proc.stdout is None:
            return
        chars: List[str] = []
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                break
            if ch in {"\n", "\r"}:
                if chars:
                    output_queue.put("".join(chars))
                    chars = []
            else:
                chars.append(ch)
        if chars:
            output_queue.put("".join(chars))

    threading.Thread(target=_enqueue_output, daemon=True).start()


def _drain_output_lines(output_queue: "queue.Queue[str]", logs: List[str]) -> List[str]:
    while True:
        try:
            line = output_queue.get_nowait()
        except queue.Empty:
            break
        logs.append(line)
    if len(logs) > 2000:
        logs = logs[-2000:]
    return logs


def _ensure_run_state() -> None:
    if "run_active" not in st.session_state:
        st.session_state["run_active"] = False
    if "run_proc" not in st.session_state:
        st.session_state["run_proc"] = None
    if "run_output_queue" not in st.session_state:
        st.session_state["run_output_queue"] = None
    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []
    if "run_return_code" not in st.session_state:
        st.session_state["run_return_code"] = None
    if "run_reported" not in st.session_state:
        st.session_state["run_reported"] = True
    if "run_saved" not in st.session_state:
        st.session_state["run_saved"] = True
    if "run_meta" not in st.session_state:
        st.session_state["run_meta"] = {}


def _start_live_run(cmd: List[str], extra_env: Dict[str, str], run_meta: Dict[str, str]) -> None:
    _ensure_run_state()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_ASCII", "1")
    env.setdefault("TQDM_NCOLS", "80")
    env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    output_queue: "queue.Queue[str]" = queue.Queue()
    _start_output_reader(proc, output_queue)

    st.session_state["run_active"] = True
    st.session_state["run_proc"] = proc
    st.session_state["run_output_queue"] = output_queue
    st.session_state["run_logs"] = []
    st.session_state["run_return_code"] = None
    st.session_state["run_reported"] = False
    st.session_state["run_saved"] = False
    st.session_state["run_meta"] = run_meta


def _stop_live_run() -> None:
    _ensure_run_state()
    proc = st.session_state.get("run_proc")
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _poll_live_run() -> None:
    _ensure_run_state()
    output_queue = st.session_state.get("run_output_queue")
    logs = st.session_state.get("run_logs", [])
    if output_queue is not None:
        logs = _drain_output_lines(output_queue, logs)
        st.session_state["run_logs"] = logs

    proc = st.session_state.get("run_proc")
    if proc is None:
        return

    if proc.poll() is not None and (output_queue is None or output_queue.empty()):
        st.session_state["run_active"] = False
        st.session_state["run_return_code"] = proc.wait()
        if not st.session_state.get("run_saved", False):
            log_save_path = RUNTIME_CONFIG_DIR / f"last_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(log_save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(st.session_state.get("run_logs", [])))
            st.session_state["run_saved"] = True


def _quote_powershell_value(value: str) -> str:
    return value.replace('"', '`"')


def build_windows_launch_command(cmd: List[str], env_vars: Dict[str, str]) -> str:
    env_parts = [
        f'$env:{k}="{_quote_powershell_value(v)}"'
        for k, v in env_vars.items()
        if v is not None and str(v).strip() != ""
    ]
    cmd_part = subprocess.list2cmdline(cmd)
    if not env_parts:
        return cmd_part
    return "; ".join(env_parts + [cmd_part])


def build_posix_launch_command(cmd: List[str], env_vars: Dict[str, str]) -> str:
    env_parts = [
        f'{k}={shlex.quote(str(v))}'
        for k, v in env_vars.items()
        if v is not None and str(v).strip() != ""
    ]
    cmd_part = " ".join(shlex.quote(str(x)) for x in cmd)
    if not env_parts:
        return cmd_part
    return " ".join(env_parts + [cmd_part])


def find_latest_run_dir(output_dir: str, dataset_type: str, set_name: str, memory_system: str) -> Optional[Path]:
    base = Path(output_dir) / dataset_type / set_name / memory_system
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("start_at_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def discover_runs(output_root: str) -> List[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    runs = [p for p in root.glob("*/*/*/start_at_*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def parse_memory_prompt(text: str) -> Dict[str, str]:
    patterns = [
        (
            r"^Context:\s*(.*?)\n\s*User:\s*(.*?)\n\s*Based on the context provided.*$",
            "context",
            "query",
        ),
        (
            r"^User Memories:\s*(.*?)\n\s*User input:\s*(.*?)\n\s*Based on the memories provided.*$",
            "context",
            "query",
        ),
        (
            r"^相关知识：\s*(.*?)\n\s*用户输入：\s*(.*?)\n\s*请根据提供的相关知识.*$",
            "context",
            "query",
        ),
        (
            r"^用户记忆：\s*(.*?)\n\s*用户输入：\s*(.*?)\n\s*请根据提供的记忆.*$",
            "context",
            "query",
        ),
    ]

    raw = text or ""
    for pattern, context_key, query_key in patterns:
        m = re.search(pattern, raw, flags=re.S)
        if m:
            return {
                context_key: m.group(1).strip(),
                query_key: m.group(2).strip(),
                "raw": raw,
            }

    return {
        "context": "",
        "query": raw.strip(),
        "raw": raw,
    }


def render_messages(messages: List[Dict[str, str]]):
    for idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        with st.chat_message(role if role in {"user", "assistant", "system"} else "assistant"):
            if role == "user":
                st.text_area(
                    "User message",
                    value=content,
                    height=180,
                    key=f"dialog_user_msg_{idx}",
                    label_visibility="collapsed",
                    disabled=True,
                )
            else:
                st.markdown(content)


def read_json_file(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_page():
    st.subheader("Run Off-Policy / On-Policy")
    _ensure_run_state()
    _poll_live_run()

    mode = st.radio("Experiment mode", ["off-policy", "on-policy"], horizontal=True)
    memory_choices = OFF_POLICY_MEMORYS if mode == "off-policy" else ON_POLICY_MEMORYS

    datasets = load_dataset_choices()

    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_type = st.selectbox("Dataset type", ["single", "domain", "task"], index=1)
    with col2:
        set_name = st.selectbox("Set name", datasets[dataset_type])
    with col3:
        memory_system = st.selectbox("Memory system", memory_choices)

    if mode == "off-policy" and memory_system == "mem0" and set_name == "Open-Domain":
        st.warning("mem0 is not supported for Open-Domain in existing scripts.")
    if mode == "off-policy" and memory_system == "mem0" and set_name == "Long-Short":
        st.warning("mem0 is not supported for Long-Short in existing scripts.")
    if memory_system == "memoryos":
        st.info("If you use memoryos, OpenAI-compatible endpoint with provider=vllm is typically more stable in this repository.")

    st.markdown("### API / Model")
    c1, c2 = st.columns(2)
    with c1:
        provider = st.selectbox("LLM provider", ["vllm", "openai"], index=0)
        llm_model = st.text_input("LLM model", value="Qwen/Qwen3-8B")
        llm_base_url = st.text_input("LLM base URL", value="http://localhost:12366/v1")
    with c2:
        llm_api_key = st.text_input("LLM API key (optional)", value="", type="password")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.1, step=0.05)
        retrieve_k = st.number_input("Retrieve k", min_value=1, max_value=50, value=5, step=1)

    st.markdown("### Embedder (for embedder_* / mem0)")
    e1, e2, e3 = st.columns(3)
    with e1:
        embedder_provider = st.selectbox("Embedder provider", ["vllm", "openai", "huggingface"], index=0)
    with e2:
        embedder_model = st.text_input("Embedder model", value="Qwen/Qwen3-Embedding-0.6B")
    with e3:
        embedder_base_url = st.text_input("Embedder base URL", value="http://localhost:12377/v1")
    embedder_dim = st.number_input("Embedder dimension", min_value=128, max_value=4096, value=1024, step=1)

    st.markdown("### Runtime")
    r1, r2, r3 = st.columns(3)
    with r1:
        output_dir = st.text_input("Output dir", value=f"{mode}/results")
    with r2:
        cache_prefix = st.text_input("Memory cache prefix", value=f"frontend_cache/{mode}/")
    with r3:
        threads = st.number_input("Threads", min_value=1, max_value=32, value=4, step=1)
    memory_bench_path = st.text_input(
        "MEMORY_BENCH_PATH (optional)",
        value=os.getenv("MEMORY_BENCH_PATH", ""),
        help="Local dataset path. Leave empty to use default HuggingFace dataset path.",
    )

    st.markdown("### Evaluation Env")
    ev1, ev2 = st.columns(2)
    with ev1:
        evaluate_base_url = st.text_input("EVALUATE_BASE_URL (optional)", value=os.getenv("EVALUATE_BASE_URL", ""))
        evaluate_model = st.text_input("EVALUATE_MODEL (optional)", value=os.getenv("EVALUATE_MODEL", ""))
        writingbench_eval_base_url = st.text_input(
            "WRITINGBENCH_EVAL_BASE_URL (optional)",
            value=os.getenv("WRITINGBENCH_EVAL_BASE_URL", os.getenv("WRITINGBENCH_VLLM_BASE_URL", "http://localhost:12388/v1")),
        )
    with ev2:
        evaluate_api_key = st.text_input("EVALUATE_API_KEY (optional)", value=os.getenv("EVALUATE_API_KEY", ""), type="password")
        writingbench_eval_provider = st.selectbox(
            "WRITINGBENCH_EVAL_PROVIDER",
            ["vllm", "openai"],
            index=0 if os.getenv("WRITINGBENCH_EVAL_PROVIDER", "vllm") != "openai" else 1,
        )
        writingbench_eval_api_key = st.text_input(
            "WRITINGBENCH_EVAL_API_KEY (optional)",
            value=os.getenv("WRITINGBENCH_EVAL_API_KEY", ""),
            type="password",
        )
        writingbench_eval_model = st.text_input(
            "WRITINGBENCH_EVAL_MODEL (optional)",
            value=os.getenv("WRITINGBENCH_EVAL_MODEL", os.getenv("WRITINGBENCH_VLLM_MODEL", "")),
            help="Override WritingBench evaluator model name/path (supports openai/vllm provider).",
        )

    step = 10
    batch_size = 100
    max_rounds = 3
    feedback_provider = provider
    feedback_llm_model = llm_model
    feedback_llm_base_url = llm_base_url
    feedback_llm_api_key = llm_api_key
    if mode == "on-policy":
        o1, o2, o3 = st.columns(3)
        with o1:
            step = st.number_input("Step", min_value=1, max_value=100, value=10, step=1)
        with o2:
            batch_size = st.number_input("Batch size", min_value=1, max_value=5000, value=100, step=1)
        with o3:
            max_rounds = st.number_input("Max rounds", min_value=1, max_value=20, value=3, step=1)

        st.markdown("### Feedback Model (on-policy)")
        f1, f2 = st.columns(2)
        with f1:
            feedback_provider = st.selectbox(
                "Feedback LLM provider",
                ["vllm", "openai"],
                index=0 if provider == "vllm" else 1,
            )
            feedback_llm_model = st.text_input("Feedback LLM model", value=llm_model)
        with f2:
            feedback_llm_base_url = st.text_input("Feedback LLM base URL", value=llm_base_url)
            feedback_llm_api_key = st.text_input("Feedback LLM API key (optional)", value=llm_api_key, type="password")

    run_active = bool(st.session_state.get("run_active", False))

    b1, b2, b3 = st.columns(3)
    with b1:
        run_clicked = st.button("Run experiment", type="primary", disabled=run_active)
    with b2:
        prepare_only_clicked = st.button("Save config & show command", type="secondary", disabled=run_active)
    with b3:
        stop_clicked = st.button("Stop experiment", type="secondary", disabled=not run_active)

    if stop_clicked:
        _stop_live_run()
        st.warning("Stopping experiment...")
        st.rerun()

    if (run_clicked or prepare_only_clicked) and mode == "off-policy" and memory_system == "mem0" and set_name in {"Open-Domain", "Long-Short"}:
        st.error("Current scripts do not support this mem0 setting. Please change set_name or memory system.")
        return

    if run_clicked or prepare_only_clicked:
        memory_cfg = build_runtime_memory_config(
            memory_system=memory_system,
            provider=provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            temperature=temperature,
            retrieve_k=int(retrieve_k),
            embedder_provider=embedder_provider,
            embedder_model=embedder_model,
            embedder_base_url=embedder_base_url,
            embedder_dim=int(embedder_dim),
        )
        memory_cfg_path = write_runtime_config(memory_cfg, f"{mode}_{memory_system}_memory")

        feedback_cfg_path = None
        if mode == "on-policy":
            feedback_cfg = build_runtime_feedback_config(
                provider=feedback_provider,
                llm_model=feedback_llm_model,
                llm_base_url=feedback_llm_base_url,
                llm_api_key=feedback_llm_api_key,
            )
            feedback_cfg_path = write_runtime_config(feedback_cfg, f"{mode}_{memory_system}_feedback")

        cmd = build_command(
            mode=mode,
            dataset_type=dataset_type,
            set_name=set_name,
            memory_system=memory_system,
            memory_config_path=memory_cfg_path,
            output_dir=output_dir,
            cache_prefix=cache_prefix,
            threads=int(threads),
            retrieve_k=int(retrieve_k),
            step=int(step),
            batch_size=int(batch_size),
            max_rounds=int(max_rounds),
            feedback_config_path=feedback_cfg_path,
        )

        extra_env = {}
        if llm_api_key:
            extra_env["OPENAI_API_KEY"] = llm_api_key
            extra_env["VLLM_API_KEY"] = llm_api_key
        if memory_bench_path.strip():
            extra_env["MEMORY_BENCH_PATH"] = memory_bench_path.strip()
        if evaluate_base_url.strip():
            extra_env["EVALUATE_BASE_URL"] = evaluate_base_url.strip()
        if evaluate_model.strip():
            extra_env["EVALUATE_MODEL"] = evaluate_model.strip()
        if evaluate_api_key.strip():
            extra_env["EVALUATE_API_KEY"] = evaluate_api_key.strip()
        extra_env["WRITINGBENCH_EVAL_PROVIDER"] = writingbench_eval_provider
        if writingbench_eval_api_key.strip():
            extra_env["WRITINGBENCH_EVAL_API_KEY"] = writingbench_eval_api_key.strip()
        if writingbench_eval_base_url.strip():
            extra_env["WRITINGBENCH_EVAL_BASE_URL"] = writingbench_eval_base_url.strip()
            extra_env["WRITINGBENCH_VLLM_BASE_URL"] = writingbench_eval_base_url.strip()
        if writingbench_eval_model.strip():
            extra_env["WRITINGBENCH_EVAL_MODEL"] = writingbench_eval_model.strip()
            extra_env["WRITINGBENCH_VLLM_MODEL"] = writingbench_eval_model.strip()

        cmd_bash = " ".join(cmd)
        cmd_ps = build_windows_launch_command(cmd, extra_env)
        cmd_sh = build_posix_launch_command(cmd, extra_env)

        st.session_state["last_cmd_bash"] = cmd_bash
        st.session_state["last_cmd_ps"] = cmd_ps
        st.session_state["last_cmd_sh"] = cmd_sh

        if prepare_only_clicked and not run_clicked:
            st.success("Configuration saved. Copy the command above to run manually.")
        else:
            _start_live_run(
                cmd,
                extra_env=extra_env,
                run_meta={
                    "output_dir": output_dir,
                    "dataset_type": dataset_type,
                    "set_name": set_name,
                    "memory_system": memory_system,
                },
            )
            st.rerun()

    if "last_cmd_bash" in st.session_state:
        st.markdown("#### Running command")
        st.code(st.session_state["last_cmd_bash"], language="bash")
        st.markdown("#### PowerShell command")
        st.code(st.session_state.get("last_cmd_ps", ""), language="powershell")
        st.markdown("#### Shell command (macOS/Linux)")
        st.code(st.session_state.get("last_cmd_sh", ""), language="bash")

    logs = st.session_state.get("run_logs", [])
    if run_active or logs:
        st.markdown("#### Live output")
        if logs:
            st.code("\n".join(logs), language="bash")
        elif run_active:
            st.code("Process started. Waiting for first output...", language="bash")
        else:
            st.code("(No output was produced.)", language="bash")

    run_return_code = st.session_state.get("run_return_code")
    if (not run_active) and run_return_code is not None and not st.session_state.get("run_reported", True):
        run_meta = st.session_state.get("run_meta", {})
        latest = find_latest_run_dir(
            run_meta.get("output_dir", output_dir),
            run_meta.get("dataset_type", dataset_type),
            run_meta.get("set_name", set_name),
            run_meta.get("memory_system", memory_system),
        )
        if run_return_code == 0:
            st.success("Experiment completed successfully.")
            if latest:
                st.info(f"Latest run directory: {latest}")
                st.session_state["latest_run_dir"] = str(latest)
        else:
            st.error("Experiment failed. Check logs above.")
        st.session_state["run_reported"] = True

    if run_active:
        st.info("Experiment is running. You can click 'Stop experiment' at any time.")
        time.sleep(0.5)
        st.rerun()


def _render_predict_record(record: Dict):
    st.markdown("### Dialogue")
    messages = record.get("messages") or []
    if messages:
        render_messages(messages)
    else:
        st.info("No messages found in this record.")

    st.markdown("### Memory-Assembled Prompt")
    if messages:
        assembled = messages[-1].get("content", "")
        parsed = parse_memory_prompt(assembled)
        st.text_area("Extracted memory context", value=parsed["context"], height=220)
        st.text_area("Extracted user query", value=parsed["query"], height=120)

    st.markdown("### Model Response")
    st.text_area("response", value=record.get("response", ""), height=180)


def results_page():
    st.subheader("Results Explorer")

    mode = st.radio("Result type", ["off-policy", "on-policy"], horizontal=True, key="result_mode")
    default_output = "off-policy/results" if mode == "off-policy" else "on-policy/results"
    output_root = st.text_input("Result output root", value=default_output)

    if "latest_run_dir" in st.session_state:
        st.caption(f"Latest run from current app session: {st.session_state['latest_run_dir']}")

    runs = discover_runs(output_root)
    if not runs:
        st.warning("No run directories found under this root.")
        return

    run_options = [str(p) for p in runs]
    selected_run = st.selectbox("Select run directory", run_options)
    run_dir = Path(selected_run)

    try:
        rel_parts = run_dir.relative_to(Path(output_root)).parts
    except ValueError:
        rel_parts = run_dir.parts
    dataset_type = rel_parts[0] if len(rel_parts) >= 1 else "unknown"
    set_name = rel_parts[1] if len(rel_parts) >= 2 else "unknown"
    memory_system = rel_parts[2] if len(rel_parts) >= 3 else "unknown"

    st.markdown("### Run Meta")
    st.json(
        {
            "mode": mode,
            "dataset_type": dataset_type,
            "set_name": set_name,
            "memory_system": memory_system,
            "run_dir": str(run_dir),
        }
    )

    if mode == "off-policy":
        summary = read_json_file(run_dir / "summary.json")
        run_cfg = read_json_file(run_dir / "run_config.json")
        predicts = read_json_file(run_dir / "predict.json") or []

        if run_cfg:
            with st.expander("Run config", expanded=False):
                st.json(run_cfg)
        if summary:
            with st.expander("Evaluate Result Summary", expanded=True):
                st.json(summary)

        if not predicts:
            st.warning("predict.json not found or empty.")
            return

        labels = [f"{i} | {row.get('dataset', 'NA')} | test_idx={row.get('test_idx', 'NA')}" for i, row in enumerate(predicts)]
        selected_label = st.selectbox("Select dialogue record", labels)
        idx = labels.index(selected_label)
        _render_predict_record(predicts[idx])

    else:
        step_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("step_")]
        if not step_dirs:
            st.warning("No step_* directories found.")
            return

        def _step_num(path: Path) -> int:
            m = re.search(r"step_(\d+)", path.name)
            return int(m.group(1)) if m else -1

        step_dirs.sort(key=_step_num)
        step_choice = st.selectbox("Step", [p.name for p in step_dirs])
        step_dir = run_dir / step_choice

        run_cfg = read_json_file(run_dir / "run_config.json")
        summary = read_json_file(step_dir / "summary.json")
        test_predicts = read_json_file(step_dir / "test_predicts.json") or []
        train_dialogs = read_json_file(step_dir / "train_dialogs.json") or []

        if run_cfg:
            with st.expander("Run config", expanded=False):
                st.json(run_cfg)
        if summary:
            with st.expander("Evaluate Result Summary", expanded=True):
                st.json(summary)

        view_type = st.radio("View", ["test_predicts", "train_dialogs"], horizontal=True)

        if view_type == "test_predicts":
            if not test_predicts:
                st.warning("test_predicts.json not found or empty.")
                return
            labels = [f"{i} | {row.get('dataset', 'NA')} | test_idx={row.get('test_idx', 'NA')}" for i, row in enumerate(test_predicts)]
            selected_label = st.selectbox("Select dialogue record", labels)
            idx = labels.index(selected_label)
            _render_predict_record(test_predicts[idx])
        else:
            if not train_dialogs:
                st.warning("train_dialogs.json not found or empty.")
                return
            labels = [
                f"{i} | {row.get('dataset', 'NA')} | train_idx={row.get('training_set_idx', 'NA')} | test_idx={row.get('test_idx', 'NA')}"
                for i, row in enumerate(train_dialogs)
            ]
            selected_label = st.selectbox("Select training dialogue", labels)
            idx = labels.index(selected_label)
            record = train_dialogs[idx]

            st.markdown("### Training Dialogue")
            render_messages(record.get("dialog", []))
            st.markdown("### Implicit Feedback")
            st.json(record.get("implicit_feedback", []))


def main():
    st.set_page_config(page_title="MemoryBench Frontend", layout="wide")
    st.title("MemoryBench Running (by THUIR)")
    st.caption("Run off-policy/on-policy, configure API models, watch logs, and inspect memory-assembled dialogues.")

    tab_run, tab_results = st.tabs(["Run", "Results"])

    with tab_run:
        run_page()

    with tab_results:
        results_page()


if __name__ == "__main__":
    main()
