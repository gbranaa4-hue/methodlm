"""Model backends for MethodLM -- the discipline is the product, not the model.

The enforced protocol (pre-registration, tools, honest ledger, ternary second witness)
is identical no matter which model generates the text. A backend only has to turn
(system, messages) into a reply string. Swap the backend, keep the method.

  local  : Qwen-3B via llama-completion.exe   -- private, offline, free (default)
  claude : Anthropic API (Claude Opus 4.8)    -- frontier reasoning; needs
           `pip install anthropic` + credentials (ANTHROPIC_API_KEY or `ant auth login`)

messages is a list of {"role": "user"|"assistant", "content": str}, first turn user.
"""
import os
import subprocess


class LocalLlama:
    kind = "local"
    label = "local Qwen-3B (offline)"

    def __init__(self, llama, gguf, threads=8, ctx=3072):
        self.llama, self.gguf, self.threads, self.ctx = llama, gguf, threads, ctx
        self.label = f"local {os.path.splitext(os.path.basename(gguf))[0]} (gguf)"

    def generate(self, system, messages, max_tokens=230, temp=0.3):
        parts = [f"<|im_start|>system\n{system}<|im_end|>"]
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)
        p = subprocess.run([self.llama, "-m", self.gguf, "-p", prompt, "-n", str(max_tokens),
                            "-t", str(self.threads), "-c", str(self.ctx), "--special",
                            "-no-cnv", "--temp", str(temp)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.stdout.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()


class ClaudeAPI:
    kind = "claude"

    def __init__(self, model="claude-opus-4-8"):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("claude backend needs the SDK:  pip install anthropic")
        self.model = model
        self.label = f"Claude ({model})"
        try:
            self._client = anthropic.Anthropic()          # resolves key/profile from env
            self._client.models.list(limit=1)             # validate credentials up front
        except Exception as e:
            raise SystemExit(f"Claude unavailable (need ANTHROPIC_API_KEY): {str(e)[:70]}")

    def generate(self, system, messages, max_tokens=230, temp=None):
        # Opus 4.8: system is separate; temperature/thinking omitted (rejected on 4.8).
        resp = self._client.messages.create(
            model=self.model, max_tokens=max(max_tokens, 512), system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages])
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class HFModel:
    """A local Hugging Face causal LM (torch) -- e.g. the method-baked Qwen-0.5B."""
    kind = "hf"

    def __init__(self, path, label=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        self.label = label or f"HF ({os.path.basename(str(path))})"

    def generate(self, system, messages, max_tokens=230, temp=0.0):
        chat = [{"role": "system", "content": system}] + list(messages)
        enc = self.tok.apply_chat_template(chat, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        with self._torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def get_model(name, here):
    name = (name or "local").lower()
    llama = os.path.join(here, "bin", "llama-completion.exe")
    if name in ("local", "qwen", "llama", "qwen3b", "3b"):
        return LocalLlama(llama, os.path.join(here, "qwen3b.gguf"))
    if name in ("qwen05b", "qwen0.5b", "0.5b", "small"):
        return LocalLlama(llama, os.path.join(here, "qwen.gguf"))
    if name in ("baked", "method-baked", "baked05b"):
        return HFModel(os.path.join(here, "method_baked_v2"), "method-baked Qwen-0.5B")
    frontier = {"claude": "claude-opus-4-8", "opus": "claude-opus-4-8",
                "sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5", "api": "claude-opus-4-8"}
    if name in frontier:
        return ClaudeAPI(frontier[name])
    raise SystemExit(f"unknown model backend '{name}' -- use: qwen3b|qwen05b|baked|opus|sonnet|haiku")
