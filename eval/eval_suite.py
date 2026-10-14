"""Usage: python eval_suite.py MODEL_CHECKPOINT blimp,arc_easy,hellaswag,piqa,nemotron_bpb

BLiMP reports pairwise accuracy: good-sentence BPB must be lower than bad-sentence
BPB. Ties count as wrong. This byte-normalized metric differs from standard BLiMP.
ARC-Easy selects the candidate with the lowest answer-only BPB.
HellaSwag selects the candidate with the lowest ending-only BPB.
PIQA selects the solution with the lowest answer-only BPB.
All four downstream tasks also report mean_bpb_margin (higher is better):
mean incorrect-answer BPB minus correct-answer BPB, averaged over examples.
Dolma reports corpus BPB on the checkpoint's training-source holdout sample.
Requires torch, transformers, datasets, and tqdm in the training environment.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import config as datasets_config
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import constants as hub_constants
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class NativeMixtureScorer:
    """Score trainer_mixture checkpoints with their saved tokenizer heads."""

    def __init__(self, model, tokenizers, eos_ids, prepare_batch, metadata):
        self.model = model.eval()
        self.tokenizers = tokenizers
        self.eos_ids = eos_ids
        self.prepare_batch = prepare_batch
        self.metadata = metadata

    def encode_batch(self, texts, prefixes=None):
        if not texts:
            raise ValueError('Cannot score an empty batch')
        if prefixes is not None and len(prefixes) != len(texts):
            raise ValueError('Each text must have one prefix')
        prefixes = [''] * len(texts) if prefixes is None else prefixes
        # Match training routing: tokenize the complete text, breaking ties by
        # the lowest saved head index. Benchmark prefix masking still uses
        # separate prefix/answer tokenization, just as the HF scoring path does.
        full_texts = [prefix + text for prefix, text in zip(prefixes, texts)]
        lengths = [[len(item.ids) for item in tokenizer.encode_batch(
            full_texts, add_special_tokens=False)] for tokenizer in self.tokenizers]
        selected = [min(range(len(lengths)), key=lambda head: lengths[head][row])
                    for row in range(len(texts))]
        batch = {'input_ids': [], 'labels': [], 'head_ids': [], 'seq_lengths': []}
        byte_counts = []
        for text, prefix, head in zip(texts, prefixes, selected):
            tokenizer = self.tokenizers[head]
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
            count = len(text.encode('utf-8'))
            if not ids or not count:
                raise ValueError('Cannot score an empty sentence')
            length = len(prefix_ids) + len(ids)
            context = self.model.shared_backbone.max_seq_len
            if length + 1 > context:
                raise ValueError(f'Input exceeds {context}-token context including EOS context and prefix: {text!r}')
            batch['input_ids'].extend([self.eos_ids[head]] + prefix_ids + ids[:-1])
            batch['labels'].extend([-100] * len(prefix_ids) + ids)
            batch['head_ids'].extend([head] * length)
            batch['seq_lengths'].append(length)
            byte_counts.append(count)
        return batch, byte_counts

    def sentence_bpb(self, texts, prefixes=None):
        batch, byte_counts = self.encode_batch(texts, prefixes)
        return [nll / (count * math.log(2))
                for nll, count in zip(self.batch_nll(batch), byte_counts)]

    @torch.inference_mode()
    def batch_nll(self, batch):
        """Return per-sequence NLL from an independent packed native forward."""
        device = self.model.vocabulary[0].weight.device
        batch = self.prepare_batch(batch, device)
        ids, labels = batch['input_ids'], batch['labels']
        embeddings = self.model.vocabulary[0].weight.new_empty(
            (ids.numel(), self.model.d_model))
        # Keep the training forward's BF16 autocast and FP32 cross entropy.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for head, positions in batch['head_positions'].items():
                embeddings.index_copy_(0, positions, F.embedding(
                    ids.index_select(0, positions), self.model.vocabulary[head].weight))
            hidden = self.model.shared_backbone(
                embeddings, batch['seq_lengths'], cu_doc_lens=batch['cu_doc_lens'],
                position_ids=batch['position_ids'])
            losses = torch.zeros(ids.numel(), dtype=torch.float32, device=device)
            for head, positions in batch['head_positions'].items():
                logits = F.linear(hidden.index_select(0, positions),
                                  self.model.vocabulary[head].weight)
                losses.index_copy_(0, positions, F.cross_entropy(
                    logits.float(), labels.index_select(0, positions),
                    ignore_index=-100, reduction='none'))
        nll = torch.stack([piece.double().sum()
                           for piece in losses.split(batch['seq_lengths'])])
        return nll.cpu().tolist()

    def document_nll(self, text):
        """Route once on the full document, then score all tokens in windows."""
        encodings = [tokenizer.encode(text, add_special_tokens=False).ids
                     for tokenizer in self.tokenizers]
        head = min(range(len(encodings)), key=lambda index: len(encodings[index]))
        ids = encodings[head]
        if not ids:
            raise ValueError('Nonempty document produced no tokens')
        context = self.model.shared_backbone.max_seq_len
        if context < 2:
            raise ValueError('Document scoring requires a context limit of at least two')
        stride = min(context - 1, max(1, context // 2))
        sequence = [self.eos_ids[head]] + ids
        batch = {'input_ids': [], 'labels': [], 'head_ids': [], 'seq_lengths': []}
        total_nll = 0.0
        for target_start in range(1, len(sequence), stride):
            end = min(target_start + stride, len(sequence))
            begin = max(0, end - context)
            inputs = sequence[begin:end - 1]
            labels = sequence[begin + 1:end]
            masked = target_start - begin - 1
            labels[:masked] = [-100] * masked
            batch['input_ids'].extend(inputs)
            batch['labels'].extend(labels)
            batch['head_ids'].extend([head] * len(inputs))
            batch['seq_lengths'].append(len(inputs))
            # Four windows bounds logits memory and amortizes small documents'
            # multi-window forwards without crossing document attention bounds.
            if len(batch['seq_lengths']) == 4 or end == len(sequence):
                total_nll += math.fsum(self.batch_nll(batch))
                batch = {'input_ids': [], 'labels': [], 'head_ids': [], 'seq_lengths': []}
        return total_nll, len(ids)


def load_native_checkpoint(checkpoint, device):
    """Load a completed local trainer_mixture checkpoint, including global-32k."""
    from tokenizers import Tokenizer

    checkpoint = Path(checkpoint).resolve()
    marker = json.loads((checkpoint / 'complete.json').read_text())
    state_path = checkpoint / 'state.pt'
    if state_path.stat().st_size != int(marker['state_bytes']):
        raise ValueError(f'Native checkpoint size does not match completion marker: {checkpoint}')
    # These are trusted local training checkpoints, containing optimizer/RNG
    # objects as well as tensors. mmap avoids reading unused optimizer tensors.
    state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
    if state.get('format_version') != 1:
        raise ValueError('Unsupported native checkpoint format')
    step = int(state['progress']['step'])
    directory_step = re.fullmatch(r'step(\d+)', checkpoint.name)
    if (not marker.get('identity') or marker['identity'] != state['identity']
            or int(marker['step']) != step
            or (directory_step is not None and int(directory_step[1]) != step)):
        raise ValueError(f'Native checkpoint identity/step does not match completion marker: {checkpoint}')
    cfg = state['config']
    saved_tokenizers = json.loads((checkpoint.parent / 'tokenizers.json').read_text())
    paths, hashes = saved_tokenizers['paths'], saved_tokenizers['hashes']
    if not paths or len(paths) != len(hashes) or len(paths) != len(state['model']['vocabulary']):
        raise ValueError('Saved tokenizer manifest and checkpoint heads disagree')
    tokenizers, eos_ids = [], []
    vocab_size = int(state['model']['vocabulary'][0].shape[0])
    for head, (filename, expected_hash) in enumerate(zip(paths, hashes)):
        path = Path(filename)
        # /data is local to the training host; shared /scratch remains unchanged.
        if checkpoint.parts[:2] == ('/', 'net') and str(path).startswith('/data/'):
            path = Path(*checkpoint.parts[:3]) / path.relative_to('/')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected_hash:
            raise ValueError(f'Saved tokenizer SHA256 mismatch: {path}')
        tokenizer = Tokenizer.from_file(str(path))
        eos = tokenizer.token_to_id(cfg.get('eos_token', '<|endoftext|>'))
        if (eos is None or tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size
                or state['model']['vocabulary'][head].shape[0] != vocab_size):
            raise ValueError(f'Saved tokenizer vocabulary/EOS does not match checkpoint: {path}')
        tokenizers.append(tokenizer)
        eos_ids.append(eos)
    # Import the trainer and EMO implementation from the checkpoint's own host.
    model_dir = checkpoint.parents[2]
    sys.path[:0] = [str(model_dir), str(model_dir / 'EMO' / 'src')]
    spec = importlib.util.spec_from_file_location(
        '_eval_native_trainer_mixture', model_dir / 'trainer_mixture.py')
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    model = trainer.MixtureLM(
        cfg, vocab_size=vocab_size, num_heads=len(tokenizers), device=device,
        backend='flash_2' if torch.device(device).type == 'cuda' else 'torch', compile_model=False)
    trainer._load_model_state(model, state['model'])
    model.assert_tied_weights()
    metadata = {'checkpoint': str(checkpoint), 'checkpoint_step': step,
                'checkpoint_format': 'trainer_mixture_v1',
                'tokenizer_heads': len(tokenizers),
                'routing': 'shortest_full_text_lowest_index_tie',
                'routing_text': 'full_document_or_prefix_plus_candidate',
                'tokenizer_hashes': hashes}
    return NativeMixtureScorer(model, tokenizers, eos_ids, trainer.prepare_packed_batch, metadata)


@torch.inference_mode()
def sentence_bpb(model, tokenizer, texts, prefixes=None):
    """Score text tokens, optionally conditioned on separately tokenized prefixes.

    No terminal EOS is added. Padding never contributes to the loss or byte count.
    Refuse overlength inputs instead of silently truncating benchmark sentences.
    """
    if isinstance(model, NativeMixtureScorer):
        return model.sentence_bpb(texts, prefixes)
    start = tokenizer.bos_token_id
    if start is None:
        start = tokenizer.eos_token_id
    if start is None:
        raise ValueError('Tokenizer needs a BOS or EOS token for sentence scoring')
    encoded = tokenizer(texts, add_special_tokens=False)['input_ids']
    if prefixes is not None and len(prefixes) != len(texts):
        raise ValueError('Each text must have one prefix')
    prefix_ids = (tokenizer(prefixes, add_special_tokens=False)['input_ids']
                  if prefixes is not None else [[] for _ in texts])
    byte_counts = [len(text.encode('utf-8')) for text in texts]
    context = min(model.config.max_position_embeddings, tokenizer.model_max_length)
    for text, ids, prefix, count in zip(texts, encoded, prefix_ids, byte_counts):
        if not ids or not count:
            raise ValueError('Cannot score an empty sentence')
        if len(prefix) + len(ids) + 1 > context:
            raise ValueError(f'Input exceeds {context}-token context including BOS and prefix: {text!r}')
    device = next(model.parameters()).device
    width = max(len(prefix) + len(ids) for prefix, ids in zip(prefix_ids, encoded))
    inputs = torch.full((len(texts), width), start, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, -100)
    attention = torch.zeros_like(inputs)
    for row, (prefix, ids) in enumerate(zip(prefix_ids, encoded)):
        length = len(prefix) + len(ids)
        inputs[row, :length] = torch.tensor([start] + prefix + ids[:-1], device=device)
        labels[row, len(prefix):length] = torch.tensor(ids, device=device)
        attention[row, :length] = 1
    logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels,
                             reduction='none', ignore_index=-100)
    nll = losses.double().sum(dim=1)
    denominator = torch.tensor(byte_counts, dtype=torch.float64, device=device) * math.log(2)
    return (nll / denominator).cpu().tolist()


def bpb_margin(scores, correct_index):
    """Mean incorrect-option BPB minus gold BPB for one example."""
    if len(scores) < 2 or not 0 <= correct_index < len(scores):
        raise ValueError('BPB margin requires at least two options and a valid gold index')
    return (math.fsum(score for index, score in enumerate(scores) if index != correct_index)
            / (len(scores) - 1) - scores[correct_index])


@contextmanager
def offline_datasets():
    """Probe prepared Arrow datasets without making Hub metadata requests."""
    flags = [(datasets_config, 'HF_DATASETS_OFFLINE'),
             (datasets_config, 'HF_HUB_OFFLINE'), (hub_constants, 'HF_HUB_OFFLINE')]
    previous = [(module, key, getattr(module, key)) for module, key in flags]
    try:
        for module, key in flags:
            setattr(module, key, True)
        yield
    finally:
        for module, key, value in previous:
            setattr(module, key, value)


def datasets_offline():
    return (datasets_config.HF_DATASETS_OFFLINE or datasets_config.HF_HUB_OFFLINE
            or hub_constants.HF_HUB_OFFLINE)


@contextmanager
def dataset_hub_lock():
    # Every job shares this Weka cache. Serialize cache misses, not GPU scoring.
    root = Path(datasets_config.HF_DATASETS_CACHE)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.tokenizer_eval_hub.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def hub_retry(operation):
    """Honor Hub throttling while allowing nontransient failures to surface."""
    for attempt in range(4):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, 'response', None)
            if attempt == 3 or getattr(response, 'status_code', None) not in (429, 500, 502, 503, 504):
                raise
            header = response.headers.get('Retry-After', '')
            match = re.search(r'Retry after (\d+) seconds', str(error), re.IGNORECASE)
            delay = float(header) if header.isdigit() else float(match[1]) if match else 30 * 2 ** attempt
            print(f'Hub temporarily unavailable; retrying in {delay + 2:g}s', flush=True)
            time.sleep(delay + 2)


def load_eval_dataset(*args, **kwargs):
    """Reuse the exact cached benchmark before considering an online download."""
    def cached():
        with offline_datasets():
            return load_dataset(*args, **kwargs)
    try:
        return cached()
    except (ConnectionError, FileNotFoundError, ValueError):
        if datasets_offline():
            raise
    with dataset_hub_lock():
        # Another evaluator may have populated this subset while we waited.
        try:
            return cached()
        except (ConnectionError, FileNotFoundError, ValueError):
            return hub_retry(lambda: load_dataset(*args, **kwargs))


def blimp_configs():
    """Require all 67 subsets, preferring the complete shared local cache."""
    # Offline Hub discovery otherwise returns only the most recently cached subset.
    cache = Path(datasets_config.HF_DATASETS_CACHE) / 'nyu-mll___blimp'
    configs = sorted({json.loads(path.read_text())['config_name']
                      for path in cache.glob('*/*/*/dataset_info.json')})
    if len(configs) != 67 and not datasets_offline():
        with dataset_hub_lock():
            configs = sorted(hub_retry(lambda: get_dataset_config_names('nyu-mll/blimp')))
    if len(configs) != 67:
        raise ValueError(f'Expected all 67 BLiMP subsets; found {len(configs)}')
    return configs


def eval_blimp(model, tokenizer):
    """Count pairs with BPB(good) < BPB(bad) across all BLiMP configurations."""
    correct = 0
    pairs = 0
    margin_sum = 0.0
    configs = blimp_configs()
    for config in tqdm(configs, desc='BLiMP subsets'):
        dataset = load_eval_dataset('nyu-mll/blimp', config, split='train')
        # Eight pairs per batch bounds the full-vocabulary logits allocation.
        for offset in range(0, len(dataset), 8):
            batch = dataset[offset:offset + 8]
            texts = [text for pair in zip(batch['sentence_good'], batch['sentence_bad'])
                     for text in pair]
            scores = sentence_bpb(model, tokenizer, texts)
            correct += sum(good < bad for good, bad in zip(scores[::2], scores[1::2]))
            margin_sum += math.fsum(bad - good for good, bad in zip(scores[::2], scores[1::2]))
            pairs += len(scores) // 2
    if not pairs:
        raise ValueError('BLiMP dataset contains no pairs')
    return {'accuracy': correct / pairs, 'correct': correct, 'pairs': pairs,
            'mean_bpb_margin': margin_sum / pairs}


def eval_arc_easy(model, tokenizer):
    """Choose the answer with lowest conditional BPB on the ARC-Easy test split."""
    dataset = load_eval_dataset('allenai/ai2_arc', 'ARC-Easy', split='test')
    correct = total = 0
    margin_sum = 0.0
    for example in tqdm(dataset, desc='ARC-Easy questions'):
        prompt = f"Question: {example['question']}\nAnswer:"
        choices = example['choices']
        # The leading separator space belongs to the scored answer and byte count.
        answers = [' ' + text for text in choices['text']]
        scores = sentence_bpb(model, tokenizer, answers, prefixes=[prompt] * len(answers))
        prediction = min(range(len(scores)), key=scores.__getitem__)
        gold = choices['label'].index(example['answerKey'])
        correct += prediction == gold
        margin_sum += bpb_margin(scores, gold)
        total += 1
    if not total:
        raise ValueError('ARC-Easy dataset contains no questions')
    return {'accuracy': correct / total, 'correct': correct, 'total': total,
            'mean_bpb_margin': margin_sum / total}


def eval_hellaswag(model, tokenizer):
    """Score a fixed random sample of 1,000 HellaSwag validation examples."""
    # Match lm-evaluation-harness's prompt construction and WikiHow cleanup.
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py
    def clean(text):
        text = text.strip().replace(' [title]', '. ')
        return re.sub(r'\[.*?\]', '', text).replace('  ', ' ')

    dataset = load_eval_dataset('Rowan/hellaswag', split='validation')
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    correct = total = 0
    margin_sum = 0.0
    for example in tqdm(dataset, desc='HellaSwag examples'):
        context = example['ctx_a'] + ' ' + example['ctx_b'].capitalize()
        prompt = clean(example['activity_label'] + ': ' + context)
        endings = [' ' + clean(ending) for ending in example['endings']]
        scores = sentence_bpb(model, tokenizer, endings, prefixes=[prompt] * len(endings))
        prediction = min(range(len(scores)), key=scores.__getitem__)
        gold = int(example['label'])
        correct += prediction == gold
        margin_sum += bpb_margin(scores, gold)
        total += 1
    if not total:
        raise ValueError('HellaSwag dataset contains no examples')
    return {'accuracy': correct / total, 'correct': correct, 'total': total,
            'mean_bpb_margin': margin_sum / total}


def eval_piqa(model, tokenizer):
    """Score a fixed random sample of 1,000 PIQA validation examples."""
    dataset = load_eval_dataset('baber/piqa', split='validation')
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    correct = total = 0
    margin_sum = 0.0
    for example in tqdm(dataset, desc='PIQA questions'):
        prompt = f"Question: {example['goal']}\nAnswer:"
        answers = [' ' + example['sol1'], ' ' + example['sol2']]
        scores = sentence_bpb(model, tokenizer, answers, prefixes=[prompt] * 2)
        prediction = min(range(len(scores)), key=scores.__getitem__)
        gold = int(example['label'])
        correct += prediction == gold
        margin_sum += bpb_margin(scores, gold)
        total += 1
    if not total:
        raise ValueError('PIQA dataset contains no questions')
    return {'accuracy': correct / total, 'correct': correct, 'total': total,
            'mean_bpb_margin': margin_sum / total}


@torch.inference_mode()
def document_nll(model, tokenizer, text):
    """Score every document token once, with overlapping context for long documents."""
    if isinstance(model, NativeMixtureScorer):
        return model.document_nll(text)
    start = tokenizer.bos_token_id
    if start is None:
        start = tokenizer.eos_token_id
    if start is None:
        raise ValueError('Tokenizer needs a BOS or EOS token for document scoring')
    # Disable the tokenizer length warning: windows below enforce the context limit.
    ids = tokenizer(text, add_special_tokens=False, verbose=False)['input_ids']
    if not ids:
        raise ValueError('Nonempty document produced no tokens')
    context = min(model.config.max_position_embeddings, tokenizer.model_max_length)
    if context < 2:
        raise ValueError('Document scoring requires a context limit of at least two')
    stride = min(context - 1, max(1, context // 2))
    sequence = [start] + ids
    device = next(model.parameters()).device
    total_nll = 0.0
    for target_start in range(1, len(sequence), stride):
        end = min(target_start + stride, len(sequence))
        begin = max(0, end - context)
        inputs = torch.tensor([sequence[begin:end - 1]], device=device)
        labels = torch.tensor([sequence[begin + 1:end]], device=device)
        # Earlier tokens in this window are context only, already scored previously.
        labels[:, :target_start - begin - 1] = -100
        logits = model(input_ids=inputs, use_cache=False).logits
        losses = F.cross_entropy(logits.float().transpose(1, 2), labels,
                                 reduction='none', ignore_index=-100)
        total_nll += losses.double().sum().item()
    return total_nll, len(ids)


DOLMA_DATA_FILE = Path(__file__).resolve().parent.parent / 'data' / 'dolma3_250gb.jsonl'
DOLMA_SAMPLE_DIR = Path(__file__).resolve().parent / '.cache' / 'dolma_samples'
DOLMA_HELDOUT_FILE = None


def dolma_data_file(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    conversion = checkpoint / 'conversion.json'
    if not conversion.is_file():
        return DOLMA_DATA_FILE
    source = Path(json.loads(conversion.read_text())['source_checkpoint'])
    # A remotely mounted export keeps native paths from its original host.
    if checkpoint.parts[:2] == ('/', 'net') and str(source).startswith('/data/'):
        source = Path(*checkpoint.parts[:3]) / source.relative_to('/')
    configured = Path(json.loads((source / 'config.json').read_text())['data_file'])
    if not configured.is_absolute():
        # Training resolves data_file relative to train/model/config, not eval's cwd.
        return (source.parents[2] / 'config' / configured).resolve()
    if source.parts[:2] == ('/', 'net'):
        return Path(*source.parts[:3]) / configured.relative_to('/')
    return configured


def dolma_sample():
    """Cache a uniform reservoir sample of 1,000 nonempty held-out documents."""
    if DOLMA_HELDOUT_FILE is not None:
        documents = json.loads(DOLMA_HELDOUT_FILE.read_text())['documents']
        if len(documents) != 1000 or len({doc['index'] for doc in documents}) != 1000:
            raise ValueError('Expected 1,000 unique shared held-out documents')
        return documents
    stat = DOLMA_DATA_FILE.stat()
    identity = [str(DOLMA_DATA_FILE.resolve()), stat.st_size, stat.st_mtime_ns,
                'heldout-0.01-seed42-reservoir-seed42-n1000-v1']
    fingerprint = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    DOLMA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    cache = DOLMA_SAMPLE_DIR / f'{fingerprint}.json'
    # Array tasks share one sample and do not independently scan the large source.
    with cache.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            return json.loads(cache.read_text())['documents']
        split_rng, sample_rng = random.Random(42), random.Random(42)
        sample, seen = [], 0
        with DOLMA_DATA_FILE.open('rb') as source:
            with tqdm(total=stat.st_size, desc='Sampling Dolma (one-time scan)',
                      unit='B', unit_scale=True) as progress:
                for index, line in enumerate(source):
                    progress.update(len(line))
                    if split_rng.random() >= 0.01:
                        continue
                    text = json.loads(line)['text']
                    if not text:
                        continue
                    seen += 1
                    if len(sample) < 1000:
                        sample.append({'index': index, 'text': text})
                    else:
                        slot = sample_rng.randrange(seen)
                        if slot < 1000:
                            sample[slot] = {'index': index, 'text': text}
        if len(sample) < 1000:
            raise ValueError(f'Need 1,000 nonempty held-out documents; found {seen}')
        current = DOLMA_DATA_FILE.stat()
        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Dolma source changed during sampling')
        sample.sort(key=lambda doc: doc['index'])
        with tempfile.NamedTemporaryFile(mode='w', dir=DOLMA_SAMPLE_DIR, delete=False) as tmp:
            json.dump({'source': identity, 'eligible_documents': seen, 'documents': sample}, tmp)
        Path(tmp.name).replace(cache)
        return sample


def eval_dolma_bpb(model, tokenizer):
    """Report corpus BPB on a shared random sample of 1,000 held-out documents."""
    total_nll = 0.0
    documents = total_bytes = tokens = empty_documents = 0
    for document in tqdm(dolma_sample(), desc='Dolma held-out documents', unit='doc'):
        text = document['text']
        nll, count = document_nll(model, tokenizer, text)
        total_nll += nll
        tokens += count
        total_bytes += len(text.encode('utf-8'))
        documents += 1
    if not total_bytes:
        raise ValueError('Dolma holdout contains no nonempty text')
    return {'bpb': total_nll / (math.log(2) * total_bytes),
            'data_file': str(DOLMA_DATA_FILE),
            'heldout_file': str(DOLMA_HELDOUT_FILE) if DOLMA_HELDOUT_FILE else None,
            'documents': documents, 'bytes': total_bytes, 'tokens': tokens,
            'empty_documents': empty_documents}


NEMOTRON_LANGUAGES = ('en', 'fr', 'de', 'zh', 'es', 'ja', 'pt', 'it', 'ru', 'pl', 'nl', 'ar')
NEMOTRON_DATA_FILE = (Path(__file__).resolve().parent.parent / 'train' / 'tokenizer'
                      / 'multilingual-compression' / 'mixture-sweep-18' / 'eval.jsonl')


def eval_nemotron_bpb(model, tokenizer):
    """Score the existing fixed Nemotron evaluation texts separately by language."""
    manifest = json.loads(NEMOTRON_DATA_FILE.with_name('manifest.json').read_text())
    expected = manifest['eval']
    if any(language not in expected for language in NEMOTRON_LANGUAGES):
        raise ValueError('A requested language is missing from the Nemotron manifest')
    stats = {language: {'nll': 0.0, 'documents': 0, 'bytes': 0, 'tokens': 0}
             for language in NEMOTRON_LANGUAGES}
    hashes = {language: hashlib.sha256() for language in NEMOTRON_LANGUAGES}
    seen = set()
    before = NEMOTRON_DATA_FILE.stat()
    total = sum(expected[language]['documents'] for language in NEMOTRON_LANGUAGES)
    with NEMOTRON_DATA_FILE.open(encoding='utf-8') as source:
        with tqdm(total=total, desc='Nemotron multilingual BPB', unit='doc') as progress:
            for line in source:
                row = json.loads(line)
                language = row['language']
                if language not in stats:
                    continue
                if row.get('split') != 'eval' or row.get('source_row_group') != 0:
                    raise ValueError('Nemotron input contains a non-evaluation document')
                key = (language, row['source_id'])
                if key in seen:
                    raise ValueError(f'Duplicate Nemotron document: {key}')
                seen.add(key)
                text = row['text']
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('Nemotron evaluation requires nonempty text')
                raw = text.encode('utf-8')
                hashes[language].update(len(raw).to_bytes(8, 'big'))
                hashes[language].update(raw)
                nll, tokens = document_nll(model, tokenizer, text)
                stat = stats[language]
                stat['nll'] += nll
                stat['documents'] += 1
                stat['bytes'] += len(raw)
                stat['tokens'] += tokens
                progress.update(1)
    after = NEMOTRON_DATA_FILE.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Nemotron evaluation source changed while scoring')
    for language, stat in stats.items():
        if (stat['documents'] != expected[language]['documents']
                or stat['bytes'] != expected[language]['utf8_bytes']):
            raise ValueError(f'Nemotron sample differs from manifest for {language}')
        stat['bpb'] = stat.pop('nll') / (math.log(2) * stat['bytes'])
        stat['text_sha256'] = hashes[language].hexdigest()
    return {'data_file': str(NEMOTRON_DATA_FILE), 'by_language': stats,
            'macro_mean_bpb': math.fsum(stat['bpb'] for stat in stats.values()) / len(stats)}


EVALS = {'blimp': eval_blimp, 'arc_easy': eval_arc_easy,
         'hellaswag': eval_hellaswag, 'piqa': eval_piqa, 'dolma_bpb': eval_dolma_bpb,
         'nemotron_bpb': eval_nemotron_bpb}


def main():
    global DOLMA_DATA_FILE, DOLMA_SAMPLE_DIR, DOLMA_HELDOUT_FILE, NEMOTRON_DATA_FILE, NEMOTRON_LANGUAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_checkpoint', help='Local Hugging Face or native trainer_mixture checkpoint directory')
    parser.add_argument('evals', help='Comma-separated eval names: ' + ','.join(EVALS))
    parser.add_argument('--nemotron-file', type=Path, default=NEMOTRON_DATA_FILE)
    parser.add_argument('--nemotron-languages', default=','.join(NEMOTRON_LANGUAGES))
    parser.add_argument('--dolma-sample-dir', type=Path, default=DOLMA_SAMPLE_DIR)
    parser.add_argument('--dolma-heldout-file', type=Path,
                        help='Shared evaluation documents excluded from every training pool')
    parser.add_argument('--output-dir', type=Path, help='Atomically save each completed evaluation')
    args = parser.parse_args()
    DOLMA_SAMPLE_DIR = args.dolma_sample_dir.resolve()
    DOLMA_HELDOUT_FILE = args.dolma_heldout_file.resolve() if args.dolma_heldout_file else None
    NEMOTRON_DATA_FILE = args.nemotron_file.resolve()
    NEMOTRON_LANGUAGES = tuple(args.nemotron_languages.split(','))
    if not all(NEMOTRON_LANGUAGES) or len(set(NEMOTRON_LANGUAGES)) != len(NEMOTRON_LANGUAGES):
        parser.error('Nemotron languages must be nonempty and unique')
    names = [name.strip() for name in args.evals.split(',')]
    if any(name not in EVALS for name in names):
        parser.error(f'Available evals: {", ".join(EVALS)}')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    native = (Path(args.model_checkpoint) / 'state.pt').is_file()
    metadata = {}
    if native:
        if 'dolma_bpb' in names:
            parser.error('Native trainer_mixture runs have no held-out Dolma split; select downstream evals only')
        model = load_native_checkpoint(args.model_checkpoint, device)
        tokenizer = None
        metadata = model.metadata
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, local_files_only=True)
        if 'dolma_bpb' in names:
            DOLMA_DATA_FILE = dolma_data_file(args.model_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint, local_files_only=True).to(device).eval()
    for name in names:
        metrics = EVALS[name](model, tokenizer)
        result = {'eval': name, 'model_checkpoint': str(Path(args.model_checkpoint).resolve()),
                  **metadata, **metrics}
        encoded = json.dumps(result, allow_nan=False)
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=args.output_dir, delete=False) as tmp:
                tmp.write(encoded + '\n')
            Path(tmp.name).replace(args.output_dir / f'{name}.json')
        print(encoded, flush=True)


if __name__ == '__main__':
    main()
