"""Document repacking adapted from allenai/EMO sewonm/icsl at 6ef92fc3.

Only the dynamic document permutation/gathering implementation is ported. This
adapter consumes this project's unpadded, single-file token streams, without the
upstream subset-manifest machinery. Epoch 1 uses the original FSL read path.
"""

import gzip
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..exceptions import OLMoConfigurationError
from ..io import is_url
from .numpy_dataset import NumpyFSLDataset
from .utils import get_document_lengths, get_rng, load_array_slice_into_tensor


class NumpyFSLDynamicRepackedDataset(NumpyFSLDataset):
    """Keep epoch 1 unchanged; permute whole documents before packing later epochs.

    ``document_metadata_path`` is gzip CSV with one ``start,end`` token span per
    original document, including its trailing EOS. Boundaries must be recorded
    during tokenization, not inferred from EOS (which may occur inside text).
    The underlying token array is never rewritten. As with ordinary FSL, the
    final partial sequence is dropped, followed by any incomplete global batch.
    Metadata is only needed when entering epoch 2 or restoring a later epoch.
    """

    _REPACKING_SEED_DOMAIN = 0x44594E52  # "DYNR", identical to upstream
    VERSION = 1

    def __init__(self, *paths, document_metadata_path: str,
                 num_documents: int, document_metadata_digest: Optional[str] = None,
                 **kwargs):
        if len(paths) != 1 or is_url(str(paths[0])):
            raise OLMoConfigurationError("repack requires one local token file")
        if num_documents <= 0:
            raise OLMoConfigurationError("repack requires a positive document count")
        if kwargs.get("label_mask_paths") or kwargs.get("max_target_sequence_length"):
            raise OLMoConfigurationError("repack does not support label masks or sequence-length warmup")
        super().__init__(*paths, **kwargs)
        self.document_metadata_path = Path(document_metadata_path)
        self.num_documents = num_documents
        self.document_metadata_digest = document_metadata_digest
        self._document_offsets = None
        self._repacked_document_offsets = None
        self._repacked_cumulative_ends = None
        self._repacking_epoch = None
        self._repacking_seed = None

    @property
    def fingerprint(self) -> str:
        # Same source contents as FSL: accept an ordinary epoch-1 checkpoint
        # without ignoring fingerprint checks. Loader state separately validates
        # the repack version and exact boundary digest for later-epoch resumes.
        digest = hashlib.sha256(b"class=NumpyFSLDataset")
        for name in self.fingerprint_fields:
            digest.update(f"{name}={getattr(self, name)},".encode())
        for path, size in zip(self.paths, self.file_sizes):
            digest.update(f"path={Path(path).name},size={size},".encode())
        return digest.hexdigest()

    @property
    def repacking_epoch(self):
        return self._repacking_epoch

    def validate_document_metadata(self) -> np.ndarray:
        if self._document_offsets is not None:
            return self._document_offsets
        path = self.document_metadata_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing repack document boundaries: {path}. Before multi-epoch training, "
                f"run: python scripts/prepare_repack.py --data-dir {Path(self.paths[0]).parent}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        if self.document_metadata_digest is not None and checksum != self.document_metadata_digest:
            raise OLMoConfigurationError("repack document metadata checksum mismatch")

        def values():
            with gzip.open(path, "rt") as source:
                for row, line in enumerate(source):
                    fields = line.strip().split(",")
                    if row >= self.num_documents or len(fields) != 2:
                        raise OLMoConfigurationError("repack metadata document count/format mismatch")
                    start, end = map(int, fields)
                    if not 0 <= start < end <= self.file_sizes[0] // self.dtype(0).itemsize:
                        raise OLMoConfigurationError("invalid repack document span")
                    yield start
                    yield end

        offsets = np.fromiter(values(), dtype=np.uint64)
        if offsets.size != self.num_documents * 2:
            raise OLMoConfigurationError("repack metadata document count mismatch")
        offsets = offsets.reshape(-1, 2)
        starts, ends = offsets[:, 0], offsets[:, 1]
        if starts[0] != 0 or np.any(starts[1:] != ends[:-1]):
            raise OLMoConfigurationError("repack requires contiguous document spans starting at token 0")
        if int(ends[-1]) * self.dtype(0).itemsize != self.file_sizes[0]:
            raise OLMoConfigurationError("repack document spans do not cover the full token file")
        self.document_metadata_digest = checksum
        self._document_offsets = offsets
        return offsets

    def repack_for_epoch(self, epoch: int, *, seed: int):
        if epoch <= 0:
            raise ValueError("epoch must be at least 1")
        if self._repacking_epoch == epoch and self._repacking_seed == seed:
            return
        if epoch == 1:
            self._repacked_document_offsets = self._repacked_cumulative_ends = None
        else:
            offsets = self.validate_document_metadata()
            order = np.arange(self.num_documents, dtype=np.uint32)
            rng = get_rng(seed + epoch + self._REPACKING_SEED_DOMAIN)
            rng.shuffle(order)
            repacked_offsets = np.take(offsets, order, axis=0)
            lengths = repacked_offsets[:, 1] - repacked_offsets[:, 0]
            cumulative_ends = np.cumsum(lengths, dtype=np.uint64)
            if int(cumulative_ends[-1]) * self.dtype(0).itemsize != self.file_sizes[0]:
                raise RuntimeError("dynamic repacking changed the token count")
            self._repacked_document_offsets = repacked_offsets
            self._repacked_cumulative_ends = cumulative_ends
        self._repacking_epoch = epoch
        self._repacking_seed = seed

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._repacking_epoch is None:
            raise RuntimeError("call data_loader.reshuffle() before reading repacked data")
        if self._repacking_epoch == 1:
            return super().__getitem__(index)
        index = int(index)
        pos_index = index if index >= 0 else len(self) + index
        if pos_index < 0 or pos_index >= len(self):
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        logical_start = pos_index * self.sequence_length
        logical_end = logical_start + self.sequence_length
        input_ids = torch.empty((self.sequence_length,), dtype=torch.long)
        write_offset = 0
        logical_pos = logical_start
        doc_index = int(np.searchsorted(self._repacked_cumulative_ends, logical_pos, side="right"))
        previous_end = 0 if doc_index == 0 else int(self._repacked_cumulative_ends[doc_index - 1])
        offset_in_doc = logical_pos - previous_end
        while logical_pos < logical_end:
            source_start, source_end = self._repacked_document_offsets[doc_index]
            source_start = int(source_start) + offset_in_doc
            take = min(int(source_end) - source_start, logical_end - logical_pos)
            input_ids[write_offset:write_offset + take] = load_array_slice_into_tensor(
                self.paths[0], source_start, source_start + take, self.dtype
            )
            write_offset += take
            logical_pos += take
            doc_index += 1
            offset_in_doc = 0

        out: Dict[str, Any] = {"input_ids": input_ids}
        if self.instance_filter_config is not None:
            out["instance_mask"] = self._validate_instance(input_ids, self.instance_filter_config)
        if self._include_instance_metadata:
            out["metadata"] = deepcopy(self._metadata[0])
        if self._generate_doc_lengths:
            out["doc_lens"] = get_document_lengths(input_ids, self.eos_token_id, bos_token_id=self.bos_token_id)
        return out

    def repacking_state(self):
        # Epoch 1 requires no sidecar and is safe to branch from old checkpoints.
        return {"version": self.VERSION, "num_documents": self.num_documents,
                "document_metadata_digest": self.document_metadata_digest}

    def load_repacking_state(self, state, epoch):
        if epoch <= 1:
            return
        self.validate_document_metadata()
        if state != self.repacking_state():
            raise RuntimeError("cannot resume a later epoch with a different repack plan")
