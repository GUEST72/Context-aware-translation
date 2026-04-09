import argparse
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import requests
from transformers import logging as transformers_logging
from initial_model import ContextAwareTranslator

def get_local_fallback():
    transformers_logging.set_verbosity_error()
    return ContextAwareTranslator()

class ContextualTranslator:
    _THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _TAGGED_TRANSLATION_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    _LATIN_RE = re.compile(r"[A-Za-z]+")
    _DIACRITICS_RE = re.compile(r"[\u064B-\u0652]")
    _SENTENCE_SPLIT_RE = re.compile(r'([.،!؟])')
    _TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, verbose: bool = False):
        self.hf_token = os.environ.get("HF_TOKEN")
        self.gh_token = os.environ.get("GITHUB_TOKEN")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.hf_url = "https://router.huggingface.co/v1/chat/completions"
        self.gh_url = "https://models.inference.ai.azure.com/chat/completions"
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.gemini_url_template = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
        self.google_free_url = "https://translate.googleapis.com/translate_a/single"
        self.mymemory_url = "https://api.mymemory.translated.net/get"
        self.hf_models = ["deepseek-ai/DeepSeek-R1", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
        self.gemini_models = ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-flash-latest"]
        self.timeout = 15
        self.min_interval = 0.5
        self.cooldown_period = 65
        self.verbose = verbose
        self._session = requests.Session()
        self._last_request_ts = 0.0
        self._cooldowns: Dict[str, float] = {}
        self._disabled_providers: Set[str] = set()
        self._cache: Dict[Tuple[str, str], str] = {}
        self._local_engine = None

    def _log(self, msg: str):
        if self.verbose: print(msg)

    def _get_local_engine(self):
        if not self._local_engine:
            self._local_engine = get_local_fallback()
        return self._local_engine

    def _is_available(self, provider: str) -> bool:
        if provider in self._disabled_providers: return False
        return time.time() >= self._cooldowns.get(provider, 0.0)

    def _trigger_cooldown(self, provider: str, duration: int = 0):
        self._cooldowns[provider] = time.time() + (duration or self.cooldown_period)
        self._log(f"[{provider.upper()}] Cooling down due to error or limit...")

    def _enforce_pacing(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _post(self, provider: str, url: str, headers: Dict, payload: Dict) -> Optional[requests.Response]:
        if not self._is_available(provider): return None
        self._enforce_pacing()
        try:
            resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
            self._last_request_ts = time.time()
            if resp.status_code == 200: return resp
            if resp.status_code in self._TRANSIENT_STATUSES: self._trigger_cooldown(provider)
            return None
        except Exception as e:
            self._log(f"[{provider.upper()}] Error: {e}")
            return None
    
    def _translate_hf(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        if not self.hf_token: return None
        for model in self.hf_models:
            resp = self._post("hf", self.hf_url, {"Authorization": f"Bearer {self.hf_token}"}, 
                             {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": max_tokens})
            if resp: return resp.json()['choices'][0]['message']['content']
        return None

    def _translate_github(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        if not self.gh_token: return None
        resp = self._post("github", self.gh_url, {"Authorization": f"Bearer {self.gh_token}"},
                         {"messages": [{"role": "user", "content": prompt}], "model": "gpt-4o-mini", "temperature": 0.1, "max_tokens": max_tokens})
        if resp: return resp.json()['choices'][0]['message']['content']
        return None

    def _translate_gemini(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        if not self.gemini_key: return None
        for model in self.gemini_models:
            url = self.gemini_url_template.format(model=model, key=self.gemini_key)
            resp = self._post("gemini", url, {}, {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}})
            if resp: return resp.json()['candidates'][0]['content']['parts'][0]['text']
        return None

    def _translate_groq(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        if not self.groq_key: return None
        resp = self._post("groq", self.groq_url, {"Authorization": f"Bearer {self.groq_key}"},
                         {"messages": [{"role": "user", "content": prompt}], "model": "llama-3.1-8b-instant", "temperature": 0.1, "max_tokens": max_tokens})
        if resp: return resp.json()['choices'][0]['message']['content']
        return None

    def _translate_google_free(self, text: str) -> Optional[str]:
        if not self._is_available("google_free"): return None
        params = {"client": "gtx", "sl": "en", "tl": "ar", "dt": "t", "q": text}
        try:
            self._enforce_pacing()
            resp = self._session.get(self.google_free_url, params=params, timeout=15)
            self._last_request_ts = time.time()
            if resp.status_code == 200: return "".join([p[0] for p in resp.json()[0] if p[0]])
            if resp.status_code == 429: self._trigger_cooldown("google_free", 120)
        except: pass
        return None
    def _translate_mymemory(self, text: str) -> Optional[str]:
        if not self._is_available("mymemory"): return None
        params = {"q": text, "langpair": "en|ar"}
        try:
            self._enforce_pacing()
            resp = self._session.get(self.mymemory_url, params=params, timeout=15)
            self._last_request_ts = time.time()
            if resp.status_code == 200: return resp.json().get('responseData', {}).get('translatedText')
            if resp.status_code == 429: self._trigger_cooldown("mymemory", 180)
        except: pass
        return None

    def _robust_extract(self, full_text_para: str, target_iso_trans: str) -> str:
        p = self._clean_output(full_text_para)
        t = self._clean_output(target_iso_trans)
        parts = self._SENTENCE_SPLIT_RE.split(p)
        sentences = []
        for i in range(0, len(parts)-1, 2):
            sentences.append((parts[i] + parts[i+1]).strip())
        if len(parts) % 2 != 0 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        if not sentences: return p
        target_words = set(t.split())
        best_score = -1.0
        best_span = p
        for i in range(len(sentences)):
            for j in range(i, len(sentences)):
                candidate = " ".join(sentences[i:j+1]).strip()
                cand_words = set(candidate.split())
                intersection = len(target_words.intersection(cand_words))
                union = len(target_words.union(cand_words))
                score = intersection / union if union > 0 else 0
                if score > best_score or (score == best_score and len(candidate) > len(best_span)):
                    best_score = score
                    best_span = candidate
        return best_span

    def translate(self, context: str, target_text: str) -> tuple[str, str]:
        cache_key = (context, target_text)
        if cache_key in self._cache: return self._cache[cache_key], "Cache"
        est_tokens = max(2048, len(target_text.split()) * 30)
        prompt = (f"System: You are an expert Arabic translator specialized in Artificial Intelligence and Deep Learning research.\n"
                  f"Your objective is to translate ONLY the 'Target' sentence into formal, professional academic Arabic script.\n\n"
                  f"STRICT DIRECTIVES:\n"
                  f"1. Translate ONLY the content of 'Target'. Do NOT add preamble, extra context, or hallucinations.\n"
                  f"2. Use pure Arabic scientific terminology for ALL technical concepts. Strictly forbid the use of transliterations or phonetic equivalents of English terms.\n"
                  f"3. Use the 'Context' ONLY to resolve linguistic ambiguity (gender, number, and domain nuance).\n"
                  f"4. Output ONLY the Arabic translation, strictly avoiding any English characters or explanations.\n"
                  f"5. Wrap the complete translation in <tr>...</tr> tags.\n\n"
                  f"Context: {context}\n"
                  f"Target: {target_text}")
        providers = [
            self._translate_hf, 
            self._translate_groq, 
            self._translate_gemini, 
            self._translate_github,
        ]
        for provider_fn in providers:
            provider_name = provider_fn.__name__.replace("_translate_", "").upper()
            res = provider_fn(prompt, max_tokens=est_tokens)
            if res:
                final = self._clean_output(res)
                if final:
                    self._cache[cache_key] = final
                    return final, provider_name
        fallbacks = [
            ("google", self._translate_google_free),
            ("mymemory", self._translate_mymemory)
        ]
        for name, fn in fallbacks:
            self._log(f"[INFO] Attempting Contextual {name.upper()}...")
            arabic_para = fn(context)
            if not arabic_para: continue
            target_iso = fn(target_text)
            if not target_iso: 
                res = arabic_para
            else:
                res = self._robust_extract(arabic_para, target_iso)
            final = self._clean_output(res)
            if final:
                self._cache[cache_key] = final
                return final, name.upper()
        self._log("[ALERT] Falling back to LOCAL mode...")
        try:
            engine = self._get_local_engine()
            arabic_para = engine.translate_block(context)
            target_iso = engine.translate_block(target_text)
            res = self._robust_extract(arabic_para, target_iso)
            final = self._clean_output(res)
            self._cache[cache_key] = final
            return final, "LOCAL"
        except Exception as e:
            return f"[Error: All levels failed. {e}]", "FAIL"

    def _clean_output(self, text: str) -> str:
        if not text: return ""
        text = self._THINK_TAG_RE.sub("", text).strip()
        matches = self._TAGGED_TRANSLATION_RE.findall(text)
        if matches:
            content = " ".join([m.strip() for m in matches if m.strip()])
        else:
            content = text
        content = self._DIACRITICS_RE.sub("", content)
        content = self._LATIN_RE.sub("", content)        
        content = re.sub(r"[^\u0600-\u06FF\s\d.,!؟؛،:()\-]+", "", content)
        return re.sub(r"\s+", " ", content).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str)
    parser.add_argument("--target", type=str)
    args = parser.parse_args()
    ctx = args.context or input("Enter CONTEXT: ").strip()
    tgt = args.target or input("Enter TARGET: ").strip()
    if ctx and tgt:
        print(ContextualTranslator().translate(ctx, tgt)[0])

if __name__ == "__main__":
    main()
