import argparse
import os
import sys
import threading
import time
from pathlib import Path

# Avoid a Transformers async-loading crash on MPS and permit CPU fallback for
# individual operations that Metal does not implement.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, TextIteratorStreamer


MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_SYSTEM_PROMPT = (
    "You will be given a question. Answer immediately using the format "
    "’Answer: [ANSWER]’ where [ANSWER] is just the number, nothing else. "
    "No explanation, no words, no reasoning, just the number."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chat with the local Qwen model, or pass a prompt for a one-shot reply."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="one-shot prompt; omit it to start an interactive chat",
    )
    parser.add_argument(
        "-n",
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"maximum response length in tokens (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; 0 gives deterministic output (default: 0)",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "system message used for the entire conversation; pass an empty string "
            "to disable it"
        ),
    )
    parser.add_argument(
        "--history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include earlier interactive turns in later prompts (default: off)",
    )
    parser.add_argument(
        "--n-filler",
        "--n_filler",
        type=int,
        default=0,
        help="number of dot filler tokens appended to each user prompt (default: 0)",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.n_filler < 0:
        parser.error("--n-filler cannot be negative")
    return args


def load_model():
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    dtype = torch.bfloat16 if use_mps else torch.float32

    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    print("Ready.")
    return model, tokenizer, device


def generate_reply(
    model,
    tokenizer,
    device: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_args = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "streamer": streamer,
    }
    if temperature > 0:
        generation_args["temperature"] = temperature

    result = []
    errors = []

    def run_generation() -> None:
        try:
            with torch.inference_mode():
                result.append(model.generate(**generation_args))
        except BaseException as error:
            errors.append(error)
            streamer.end()

    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    worker = threading.Thread(target=run_generation)
    worker.start()

    pieces = []
    print("Qwen> ", end="", flush=True)
    for piece in streamer:
        pieces.append(piece)
        print(piece, end="", flush=True)
    worker.join()
    print()

    if errors:
        raise errors[0]

    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = result[0].shape[1] - input_tokens
    rate = output_tokens / elapsed if elapsed else 0
    print(
        f"[input: {input_tokens} tokens; output: {output_tokens} tokens; "
        f"{elapsed:.1f}s; {rate:.2f} tokens/s]"
    )
    return "".join(pieces).strip()


def initial_messages(
    system_message: str | None,
    n_filler: int,
) -> list[dict[str, str]]:
    if system_message:
        if n_filler > 0:
            noun = "token" if n_filler == 1 else "tokens"
            system_message += (
                f" After the question, there will be {n_filler} filler {noun} "
                "(a sequence of dots) before you answer."
            )
        return [{"role": "system", "content": system_message}]
    return []


def add_filler(prompt: str, n_filler: int) -> str:
    if n_filler == 0:
        return prompt
    return f"{prompt} {' '.join(['.'] * n_filler)}"


def interactive_chat(model, tokenizer, device: str, args: argparse.Namespace) -> None:
    n_filler = args.n_filler
    messages = initial_messages(args.system, n_filler)
    max_new_tokens = args.max_new_tokens
    keep_history = args.history

    print(
        "Interactive chat. Commands: /help, /clear, /history on|off, "
        "/filler N, /tokens N, /quit"
    )
    print(
        f"History is {'on' if keep_history else 'off'}; token limit is "
        f"{max_new_tokens}; filler tokens: {n_filler}."
    )

    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            print("Bye.")
            return
        if prompt == "/help":
            print("/clear       forget the current conversation")
            print("/history on  include earlier turns in each prompt")
            print("/history off make every prompt independent")
            print("/filler N    set the number of appended dot tokens")
            print("/tokens N    set the maximum response length")
            print("/quit        exit")
            continue
        if prompt == "/clear":
            messages = initial_messages(args.system, n_filler)
            print("Conversation cleared.")
            continue
        if prompt.startswith("/history "):
            setting = prompt.removeprefix("/history ").strip().lower()
            if setting not in {"on", "off"}:
                print("Usage: /history on|off")
                continue
            keep_history = setting == "on"
            if not keep_history:
                messages = initial_messages(args.system, n_filler)
            print(f"History is {setting}.")
            continue
        if prompt.startswith("/filler "):
            value = prompt.removeprefix("/filler ").strip()
            try:
                requested_filler = int(value)
                if requested_filler < 0:
                    raise ValueError
            except ValueError:
                print("Usage: /filler N, where N is a non-negative integer")
                continue
            n_filler = requested_filler
            messages = initial_messages(args.system, n_filler)
            print(f"Filler tokens set to {n_filler}; conversation cleared.")
            continue
        if prompt.startswith("/tokens "):
            value = prompt.removeprefix("/tokens ").strip()
            try:
                requested_tokens = int(value)
                if requested_tokens < 1:
                    raise ValueError
            except ValueError:
                print("Usage: /tokens N, where N is a positive integer")
                continue
            max_new_tokens = requested_tokens
            print(f"Token limit is {max_new_tokens}.")
            continue
        if prompt.startswith("/"):
            print("Unknown command. Type /help for the command list.")
            continue

        user_content = add_filler(prompt, n_filler)
        request_messages = messages + [{"role": "user", "content": user_content}]
        try:
            reply = generate_reply(
                model,
                tokenizer,
                device,
                request_messages,
                max_new_tokens,
                args.temperature,
            )
        except Exception as error:
            print(f"Generation failed: {error}", file=sys.stderr)
            continue

        if keep_history:
            messages = request_messages + [{"role": "assistant", "content": reply}]


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_model()

    if args.prompt is None:
        interactive_chat(model, tokenizer, device, args)
        return

    messages = initial_messages(args.system, args.n_filler)
    messages.append(
        {"role": "user", "content": add_filler(args.prompt, args.n_filler)}
    )
    generate_reply(
        model,
        tokenizer,
        device,
        messages,
        args.max_new_tokens,
        args.temperature,
    )


if __name__ == "__main__":
    main()
