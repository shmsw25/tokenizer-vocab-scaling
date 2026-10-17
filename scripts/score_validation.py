"""Score an existing converted checkpoint on the shared document-held-out corpus."""
import json
import math
from pathlib import Path

from pipeline import digest, read_json, write_json
from validation_data import VERSION, validation_metadata


def dolma_metric(result):
    """The report and LR selection use the same identified shared-corpus score."""
    if result['identity']['validation_id'] != VERSION:
        raise ValueError('Unexpected Dolma validation identity')
    return dict(eval='dolma_bpb', bpb=result['metrics']['validation_bpb'],
        model_checkpoint=result['identity']['exported'], validation_id=VERSION,
        documents_sha256=result['identity']['documents_sha256'],
        documents=result['documents'], bytes=result['text_bytes'],
        tokens=result['stream_tokens'], predicted_tokens=result['predicted_tokens'],
        scoring=result['scoring'])


def replace_dolma_result(directory, result):
    """Replace the current Dolma result without rerunning the other benchmarks."""
    metric = dolma_metric(result)
    summary_path = directory / 'evaluation_summary.json'
    summary = read_json(summary_path) if summary_path.exists() else None
    if summary is not None and summary['checkpoint'] != result['identity']['checkpoint']:
        raise ValueError('Evaluation summary points to a different checkpoint')
    target = directory / 'evaluation' / 'dolma_bpb.json'
    if target.exists():
        previous = read_json(target)
        if previous != metric:
            # Keep immutable audit evidence outside the current results namespace.
            history = directory / 'evaluation_history' / f'dolma_{digest(target)}.json'
            if not history.exists(): write_json(history, previous)
    write_json(target, metric)
    if summary is not None:
        summary['metrics']['dolma_bpb'] = metric
        summary['shared_validation'] = result
        write_json(summary_path, summary)
    return metric


def validation_windows(values, seq):
    for start in range(0, len(values)-1, seq):
        yield start, values[start:min(start+seq+1,len(values))]


def score_stream(model, values, seq, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    nll=0.;predicted=0
    with torch.inference_mode():
        for start, window in validation_windows(values,seq):
            # One extra input token supplies the last next-token label, avoiding
            # loss of a target at each sequence boundary. Final chunk is unpadded.
            array=np.array(window,dtype=np.int64)
            x=torch.from_numpy(array[:-1]).unsqueeze(0).to(device)
            y=torch.from_numpy(array[1:]).to(device)
            logits=model(x).logits.squeeze(0)
            loss=F.cross_entropy(logits.float(),y,reduction='sum')
            if not torch.isfinite(loss):raise ValueError('Non-finite shared validation loss')
            nll+=loss.item();predicted+=y.numel()
            if start//seq%500==0:
                print(f'SHARED_VALIDATION_PROGRESS {predicted}/{len(values)-1} tokens',flush=True)
    return nll, predicted


def score_validation(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    directory = args.model_directory
    training = read_json(directory/'training_summary.json')
    dataset = getattr(args, 'dataset', 'dolma')
    if training['config'].get('pipeline', {}).get('dataset', 'dolma') != dataset:
        raise ValueError('Validation dataset does not match the trained model')
    if training['model_size'] == 'medium_large' and (
            training['config']['model'].get('block_name') != 'default'
            or training['config']['model'].get('qk_norm')):
        raise ValueError('Do not validate the superseded medium_large architecture')
    if training['data_size'] != args.data_size or training['vocab_size'] != args.vocab_size:
        raise ValueError('Validation coordinate does not match the trained model')
    converted = read_json(directory/'conversion_summary.json')
    if converted['conversion']['source_checkpoint'] != training['checkpoint']:
        raise ValueError('Converted checkpoint differs from completed training')
    exported = Path(converted['exported'])
    tokenizer_source = getattr(args, 'tokenizer_source', 'pool')
    trained_source = training['config'].get('pipeline', {}).get('tokenizer_source')
    if bool(trained_source) != (tokenizer_source == 'dolma2'):
        raise ValueError('Validation tokenizer source does not match the trained model')
    meta = validation_metadata(args.root, args.data_size, args.vocab_size, dataset=dataset,
                               tokenizer_source=tokenizer_source)
    if digest(directory/'tokenizer.json') != meta['recipe']['tokenizer_sha256']:
        raise ValueError('Model tokenizer differs from shared-validation tokenizer')
    output = directory/f'validation_{VERSION}.json'
    identity = dict(validation_id=VERSION, checkpoint=training['checkpoint'],
        exported=str(exported), documents_sha256=meta['recipe']['documents_sha256'],
        tokens_sha256=meta['sha256'], sequence_length=training['config']['seq_length'])
    if output.exists():
        result=read_json(output)
        if result['identity']!=identity:raise ValueError('Existing validation score identity changed')
        replace_dolma_result(directory, result)
        return result
    model=AutoModelForCausalLM.from_pretrained(exported,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    values=np.memmap(meta['output'],mode='r',dtype=np.uint32)
    nll, predicted = score_stream(model, values, identity['sequence_length'], 'cuda')
    # Match repository byte normalization: token-mean CE times tokens/raw UTF-8 bytes.
    ce=nll/predicted;bpb=ce*meta['tokens_per_byte']/math.log(2)
    result=dict(identity=identity,metrics={'validation_bpb':bpb,'validation_ce':ce},
        predicted_tokens=predicted,stream_tokens=len(values),documents=meta['documents'],
        text_bytes=meta['bytes'],tokenizer_sha256=meta['recipe']['tokenizer_sha256'],
        scoring='Repository byte-normalized CE: mean token NLL * stream tokens / raw UTF-8 bytes / ln(2)',
        boundary_policy='Non-overlapping context windows; one lookahead label; all stream tokens except the first are predicted')
    write_json(output,result);print('SHARED_VALIDATION_READY '+json.dumps(result),flush=True)
    replace_dolma_result(directory, result)
    return result
