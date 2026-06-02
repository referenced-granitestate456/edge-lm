"""Interactive multi-turn chat with tool use support.

Usage:
    python examples/chat.py
    python examples/chat.py --tools                     # enable built-in tools
    python examples/chat.py --tools --max-tokens 1000
    python examples/chat.py --voice                     # enable voice input (press Enter to record)
    python examples/chat.py --use-ref
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import warnings
import wave

import mlx.core as mx
import numpy as np

from mlx_vlm import stream_generate

warnings.filterwarnings("ignore", message="At least one mel filter")


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, log, pi, e.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2**10 + sqrt(144)'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python code snippet and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current directory)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (simulated). Returns a note that web access is unavailable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_tokens",
            "description": "Count the number of tokens in a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to tokenize"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text between languages (uses the model itself).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {"type": "string", "description": "Target language, e.g. 'French', 'Japanese'"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
]


def execute_tool(name: str, args: dict, tokenizer=None) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if name == "calculate":
            expr = args["expression"]
            allowed = {"__builtins__": {}, "sqrt": math.sqrt, "sin": math.sin,
                       "cos": math.cos, "tan": math.tan, "log": math.log,
                       "log10": math.log10, "pi": math.pi, "e": math.e,
                       "abs": abs, "round": round, "pow": pow}
            result = eval(expr, allowed)
            return json.dumps({"result": result})

        elif name == "get_current_time":
            now = datetime.datetime.now()
            return json.dumps({"datetime": now.isoformat(), "timezone": "local"})

        elif name == "run_python":
            result = subprocess.run(
                [sys.executable, "-c", args["code"]],
                capture_output=True, text=True, timeout=10,
            )
            return json.dumps({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})

        elif name == "read_file":
            with open(args["path"]) as f:
                content = f.read(10000)
            return json.dumps({"content": content})

        elif name == "write_file":
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return json.dumps({"status": "ok", "bytes_written": len(args["content"])})

        elif name == "list_directory":
            path = args.get("path", ".")
            entries = os.listdir(path)
            return json.dumps({"entries": sorted(entries)})

        elif name == "shell_command":
            result = subprocess.run(
                args["command"], shell=True,
                capture_output=True, text=True, timeout=10,
            )
            return json.dumps({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})

        elif name == "web_search":
            import urllib.request, urllib.parse
            query = args["query"]
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            html = resp.read().decode("utf-8")

            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            urls = re.findall(r'class="result__a"[^>]*href="(.*?)"', html)

            results = []
            for i in range(min(3, len(titles))):
                t = re.sub(r"<.*?>", "", titles[i]).strip()
                s = re.sub(r"<.*?>", "", snippets[i]).strip() if i < len(snippets) else ""
                u = urls[i] if i < len(urls) else ""
                if "uddg=" in u:
                    u = urllib.parse.unquote(re.search(r"uddg=([^&]+)", u).group(1))
                results.append({"title": t, "snippet": s, "url": u})

            # Fetch first result page
            page_text = ""
            if results and results[0]["url"]:
                try:
                    req2 = urllib.request.Request(results[0]["url"], headers={"User-Agent": "Mozilla/5.0"})
                    resp2 = urllib.request.urlopen(req2, timeout=10)
                    page_html = resp2.read().decode("utf-8", errors="replace")
                    page_text = re.sub(r"<script[^>]*>.*?</script>", "", page_html, flags=re.DOTALL)
                    page_text = re.sub(r"<style[^>]*>.*?</style>", "", page_text, flags=re.DOTALL)
                    page_text = re.sub(r"<[^>]+>", " ", page_text)
                    page_text = re.sub(r"\s+", " ", page_text).strip()[:3000]
                except Exception:
                    pass

            return json.dumps({"results": results, "first_page_content": page_text})

        elif name == "count_tokens":
            if tokenizer:
                tokens = tokenizer.encode(args["text"])
                return json.dumps({"token_count": len(tokens)})
            return json.dumps({"error": "Tokenizer not available"})

        elif name == "translate_text":
            return json.dumps({"note": "Translation will be handled by the model directly",
                             "text": args["text"], "target": args["target_language"]})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

TOOL_CALL_PATTERN = re.compile(
    r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>', re.DOTALL
)
PARAM_PATTERN = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>')


def parse_tool_calls(text: str) -> list[dict]:
    """Parse Gemma4 tool call format into structured dicts."""
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        params_str = match.group(2)
        args = {}
        for pm in PARAM_PATTERN.finditer(params_str):
            args[pm.group(1)] = pm.group(2)
        calls.append({"name": name, "arguments": args})
    return calls


# ---------------------------------------------------------------------------
# Voice recording
# ---------------------------------------------------------------------------

def _record_audio(sample_rate=16000):
    """Record from microphone until Enter is pressed. Returns path to temp wav."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice: pip install sounddevice")
        return None

    chunks = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        if not stop.is_set():
            chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        callback=callback, blocksize=1024):
        print("Recording... press Enter to stop ", end="", flush=True)
        input()
        stop.set()

    if not chunks:
        return None

    audio = np.concatenate(chunks)
    duration = len(audio) / sample_rate
    print(f"({duration:.1f}s)")

    if duration < 0.3:
        print("Too short, skipping.")
        return None

    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


def generate_audio(model, processor, tokenizer, audio_path, messages,
                   text_prompt="", tools=None, max_tokens=500):
    """Generate response from audio input with conversation context.

    Returns (response_text, cache) — cache can be used for tool follow-ups.
    """
    from mlx_vlm.utils import prepare_inputs

    content = [{"type": "audio", "audio": audio_path}]
    if text_prompt:
        content.append({"type": "text", "text": text_prompt})

    all_messages = list(messages) + [{"role": "user", "content": content}]
    tmpl_kwargs = {}
    if tools:
        tmpl_kwargs["tools"] = tools
    formatted = tokenizer.apply_chat_template(
        all_messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs,
    )
    inputs = prepare_inputs(processor, audio=[audio_path], prompts=formatted)
    input_ids = inputs["input_ids"]

    kwargs = {}
    for key in ("input_features", "input_features_mask", "audio_features",
                "audio_mask", "feature_attention_mask", "audio_feature_lengths"):
        if key in inputs and inputs[key] is not None:
            kwargs[key] = inputs[key]

    cache = model.language_model.make_cache()
    logits = model(input_ids, cache=cache, **kwargs)
    mx.eval(logits)
    if hasattr(logits, "logits"):
        logits = logits.logits

    tokens = []
    for _ in range(max_tokens):
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_token in (1, 106):
            break
        tokens.append(next_token)
        sys.stdout.write(tokenizer.decode([next_token]))
        sys.stdout.flush()
        logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        if hasattr(logits, "logits"):
            logits = logits.logits

    print()
    return tokenizer.decode(tokens), cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, messages, tools=None, max_tokens=500, max_context=8192,
             cache=None, cached_ids=None):
    kwargs = {}
    if tools:
        kwargs["tools"] = tools

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs,
    )
    ids = tokenizer.encode(prompt)

    if len(ids) > max_context:
        print(f"\n[Warning: context {len(ids)} tokens exceeds {max_context}, truncating]")
        cache = None
        cached_ids = None
        while len(ids) > max_context and len(messages) > 2:
            messages.pop(0)
            messages.pop(0)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs,
            )
            ids = tokenizer.encode(prompt)

    reuse = False
    if cache is not None and cached_ids is not None:
        n = len(cached_ids)
        if n <= len(ids) and ids[:n] == cached_ids:
            reuse = True

    if not reuse:
        cache = model.language_model.make_cache()
        new_ids = ids
    else:
        new_ids = ids[n:]

    if not new_ids:
        # Exact prefix match with nothing new to feed; nothing to generate.
        return "", cache, cached_ids

    # stream_generate extends `cache` in place and streams decoded text. Pass
    # input_ids directly (text-only) to skip prepare_inputs. The final yield
    # carries the stop token (or repeats the last token), so drop it when
    # rebuilding the exact id list for next turn's cache-prefix match.
    text_parts, tokens = [], []
    for result in stream_generate(
        model, tokenizer, "", input_ids=mx.array([new_ids], dtype=mx.int32),
        prompt_cache=cache, max_tokens=max_tokens,
    ):
        text_parts.append(result.text)
        sys.stdout.write(result.text)
        sys.stdout.flush()
        tokens.append(result.token)
    print()
    generated = tokens[:-1]
    cached_ids = ids + generated
    return "".join(text_parts), cache, cached_ids


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chat with gemma4")
    parser.add_argument("--model", type=str, default="TheStageAI/gemma-4-E2B-it")
    parser.add_argument("--use-ref", action="store_true", help="Use original HF model")
    parser.add_argument("--hf-model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--tools", action="store_true", help="Enable built-in tools")
    parser.add_argument("--voice", action="store_true", help="Enable voice input (press Enter to record)")
    parser.add_argument("--voice-prompt", type=str, default="", help="Text prompt for voice turns")
    args = parser.parse_args()

    audio_processor = None

    if args.use_ref:
        print(f"Loading {args.hf_model}...")
        from mlx_vlm import load as load_vlm
        model, processor = load_vlm(args.hf_model)
        tokenizer = processor.tokenizer
        if args.voice:
            audio_processor = processor
    else:
        print(f"Loading {args.model}...")
        from edge_lm.models.load import load, set_prefill_logits_to_keep
        model, tokenizer = load(args.model, include_audio=args.voice)
        set_prefill_logits_to_keep(model, 1)  # generation only needs last-token logits
        if args.voice:
            from transformers import AutoProcessor
            print("Loading audio processor...")
            audio_processor = AutoProcessor.from_pretrained(args.hf_model)

    tools = TOOL_DEFINITIONS if args.tools else None
    tools_label = f", {len(TOOL_DEFINITIONS)} tools" if tools else ""
    voice_label = ", voice" if args.voice else ""

    print(f"Ready (max_tokens={args.max_tokens}, max_context={args.max_context}{tools_label}{voice_label}).")
    cmds = "/clear /compact /context /tools /exit"
    if args.voice:
        cmds += " /v"
        print(f"Commands: {cmds}")
        print(f"Press Enter on empty line or type /v to record voice.\n")
    else:
        print(f"Commands: {cmds}\n")

    messages = []
    cache = None
    cached_ids = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input and not args.voice:
            continue
        if (not user_input or user_input in ("/v", "/voice")) and args.voice:
            audio_path = _record_audio()
            if audio_path is None:
                continue
            print("Model: ", end="")
            try:
                response, audio_cache = generate_audio(
                    model, audio_processor, tokenizer, audio_path, messages,
                    text_prompt=args.voice_prompt, tools=tools,
                    max_tokens=args.max_tokens,
                )
            finally:
                os.unlink(audio_path)
            label = f"[Voice] {args.voice_prompt}" if args.voice_prompt else "[Voice message]"
            messages.append({"role": "user", "content": label})

            # Handle tool calls from voice input
            tool_calls = parse_tool_calls(response) if tools else []
            if tool_calls:
                tc_formatted = [{"type": "function", "function": tc} for tc in tool_calls]
                messages.append({"role": "assistant", "tool_calls": tc_formatted})
                for tc in tool_calls:
                    print(f"  [Calling {tc['name']}({tc['arguments']})]")
                    result = execute_tool(tc["name"], tc["arguments"], tokenizer=tokenizer)
                    print(f"  [Result: {result[:200]}]")
                    messages.append({"role": "tool", "name": tc["name"], "content": result})
                # Follow-up uses text generate with tool results (no audio needed)
                print("Model: ", end="")
                followup, cache, cached_ids = generate(
                    model, tokenizer, messages, tools=tools,
                    max_tokens=args.max_tokens, max_context=args.max_context,
                )
                messages.append({"role": "assistant", "content": followup})
            else:
                messages.append({"role": "assistant", "content": response})
                cache = None
                cached_ids = None
            del audio_cache
            mx.clear_cache()
            print()
            continue
        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye!")
            break
        if user_input == "/context":
            kwargs = {"tools": tools} if tools else {}
            if messages:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
                n = len(tokenizer.encode(prompt))
            else:
                n = 0
            cached_n = len(cached_ids) if cached_ids else 0
            print(f"  {n} / {args.max_context} tokens ({100*n/args.max_context:.0f}%), {len(messages)//2} turns, {cached_n} cached\n")
            continue
        if user_input == "/clear":
            messages = []
            cache = None
            cached_ids = None
            print("Chat history cleared.\n")
            continue
        if user_input == "/tools":
            if tools:
                print("  Tools enabled:")
                for t in tools:
                    print(f"    - {t['function']['name']}: {t['function']['description']}")
            else:
                print("  Tools disabled. Use --tools flag to enable.")
            print()
            continue
        if user_input == "/compact":
            if len(messages) < 2:
                print("Nothing to compact.\n")
                continue
            print("Compacting... ", end="", flush=True)
            summary_msgs = messages + [{"role": "user", "content":
                "Summarize our entire conversation so far in a concise paragraph. "
                "Include all key topics, decisions, and facts discussed."}]
            summary, _, _ = generate(model, tokenizer, summary_msgs, tools=None,
                             max_tokens=args.max_tokens, max_context=args.max_context)
            old_turns = len(messages) // 2
            messages = [{"role": "user", "content": f"Summary of our prior conversation:\n{summary}"},
                        {"role": "assistant", "content": "Got it. How can I help?"}]
            cache = None
            cached_ids = None
            kwargs = {"tools": tools} if tools else {}
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
            new_tokens = len(tokenizer.encode(prompt))
            print(f"Compacted {old_turns} turns → {new_tokens} tokens\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Model: ", end="")
        response, cache, cached_ids = generate(
            model, tokenizer, messages, tools=tools,
            max_tokens=args.max_tokens, max_context=args.max_context,
            cache=cache, cached_ids=cached_ids,
        )

        # Handle tool calls
        tool_calls = parse_tool_calls(response) if tools else []
        if tool_calls:
            # Add assistant message with tool calls
            tc_formatted = [{"type": "function", "function": tc} for tc in tool_calls]
            messages.append({"role": "assistant", "tool_calls": tc_formatted})

            # Execute each tool and add results
            for tc in tool_calls:
                print(f"  [Calling {tc['name']}({tc['arguments']})]")
                result = execute_tool(tc["name"], tc["arguments"], tokenizer=tokenizer)
                print(f"  [Result: {result[:200]}]")
                messages.append({"role": "tool", "name": tc["name"], "content": result})

            # Generate follow-up with tool results
            print("Model: ", end="")
            followup, cache, cached_ids = generate(
                model, tokenizer, messages, tools=tools,
                max_tokens=args.max_tokens, max_context=args.max_context,
                cache=cache, cached_ids=cached_ids,
            )
            messages.append({"role": "assistant", "content": followup})
        else:
            messages.append({"role": "assistant", "content": response})
        mx.clear_cache()
        print()


if __name__ == "__main__":
    main()
