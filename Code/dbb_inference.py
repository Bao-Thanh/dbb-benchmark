"""One Ollama inference adapter for all experiments. Never fabricate model responses."""
from __future__ import annotations
import random
import time
from dataclasses import dataclass
import requests

BASE = 'http://127.0.0.1:11434'
MODELS = ('qwen3:0.6b','qwen3:1.7b','llama3.2:1b','gemma3:1b',
          'phi3:mini','smollm2:1.7b','stablelm2:1.6b','tinyllama:1.1b')

@dataclass(frozen=True)
class Settings:
    temperature: float = 0.8
    top_p: float = 1.0
    seed: int = 42
    num_predict: int = 96
    num_ctx: int = 2048
    timeout: int = 120

    def metadata(self) -> dict:
        return vars(self).copy()


def models_available() -> list[str]:
    try:
        r = requests.get(BASE + '/api/tags', timeout=3)
        r.raise_for_status()
        found = [x['name'] for x in r.json().get('models', [])]
        return found
    except (requests.RequestException, KeyError, ValueError):
        return []


def shuffled_options(options: dict[str, str], seed: int) -> tuple[dict[str, str],dict[str, str]]:
    """Returns (display letter -> content, display letter -> original source ID)."""
    sources = list(options)
    random.Random(int(seed)).shuffle(sources)
    letters = 'ABC'[:len(sources)]
    return ({letter: str(options[src]) for letter,src in zip(letters,sources)},
            {letter: src for letter,src in zip(letters,sources)})


def build_prompt(context: str, options: dict[str, str], question: str) -> str:
    parts = [f'Context:\n{context}', question.strip(), 'Options:']
    parts += [f'{letter}. {value}' for letter, value in options.items()]
    parts += ['Return exactly one option letter (A, B or C as available); no explanation.']
    return '\n\n'.join(s for s in parts if s)


def ask_ollama(model: str, prompt: str, settings: Settings) -> dict:
    started = time.perf_counter()
    try:
        resp = requests.post(BASE+'/api/chat', json={
            'model': model, 'stream': False, 'think': False,
            'messages': [
              {'role':'system','content':'Choose a single letter from the offered options. Reply with one letter only.'},
              {'role':'user','content':prompt}],
            'options': {'temperature':float(settings.temperature), 'top_p':float(settings.top_p),
                        'seed':int(settings.seed), 'num_predict':int(settings.num_predict),
                        'num_ctx':int(settings.num_ctx)},
            'keep_alive':'5m'
        }, timeout=int(settings.timeout))
        resp.raise_for_status()
        data = resp.json()
        raw = data.get('message', {}).get('content', '') or ''
        return {'raw':str(raw),'error':'','seconds':round(time.perf_counter()-started,3),
                'total_duration':data.get('total_duration'), 'done_reason':data.get('done_reason')}
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {'raw':'','error':f'{type(exc).__name__}: {str(exc)[:350]}',
                'seconds':round(time.perf_counter()-started,3), 'total_duration':None,
                'done_reason':None}
