from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from data.pipeline_constants import (
    CODE_CLEANUP_MAX_TOKENS,
    CODE_CLEANUP_MODEL_ID,
    MAX_TRAIN_SAMPLE_TOKENS,
    TRAIN_BASE_MODEL_ID,
)


@dataclass
class TrainingModelRuntime:
    model_id: str = TRAIN_BASE_MODEL_ID
    device: str | None = None

    def __post_init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        if self.device is None:
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required for token counting. "
                    "Install the training dependencies first."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "transformers and torch are required for embedding generation. "
                    "Install the training dependencies first."
                ) from exc

            model_kwargs, quantized = _build_model_kwargs(torch, self.device)

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            if not quantized:
                model.to(self.device)
            model.eval()

            self._torch = torch
            self._model = model

        return self._model, self._torch

    @property
    def tokenizer(self) -> Any:
        return self._load_tokenizer()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False)
        return "\n\n".join(message["content"] for message in messages)

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        return self.count_tokens(self.render_chat(messages))

    def embed_text(self, text: str) -> list[float]:
        model, torch = self._load_model()
        tokenizer = self.tokenizer
        input_device = _get_model_input_device(model)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TRAIN_SAMPLE_TOKENS,
        )
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized[0].detach().cpu().tolist()


@dataclass
class CleanupCodeEmbeddingRuntime:
    model_id: str = CODE_CLEANUP_MODEL_ID
    max_tokens: int = CODE_CLEANUP_MAX_TOKENS
    device: str | None = None

    def __post_init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        if self.device is None:
            self.device = self._detect_device()

    def _import_torch(self) -> Any:
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "torch is required for code cleanup embeddings. "
                    "Install the cleanup dependencies first."
                ) from exc
            self._torch = torch
        return self._torch

    def _detect_device(self) -> str:
        try:
            torch = self._import_torch()
        except RuntimeError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required for code cleanup embeddings. "
                    "Install the cleanup dependencies first."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required for code cleanup embeddings. "
                    "Install the cleanup dependencies first."
                ) from exc
            torch = self._import_torch()

            model_kwargs, quantized = _build_model_kwargs(torch, self.device)

            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            if not quantized:
                model.to(self.device)
            model.eval()

            self._torch = torch
            self._model = model

        return self._model, self._torch

    @property
    def tokenizer(self) -> Any:
        return self._load_tokenizer()

    @property
    def torch(self) -> Any:
        return self._import_torch()

    def embed_text_tensor(self, text: str) -> Any:
        model, torch = self._load_model()
        tokenizer = self.tokenizer
        input_device = _get_model_input_device(model)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError(
                    "Cleanup embedding model did not return hidden states for pooling."
                )
            hidden = hidden_states[-1]

        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized[0].detach().cpu()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_text_tensor(text).tolist()


BaseModelRuntime = TrainingModelRuntime


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_model_dtype(torch: Any) -> Any:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _build_model_kwargs(torch: Any, device: str | None) -> tuple[dict[str, Any], bool]:
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    quantized = False

    if device and device.startswith("cuda"):
        model_kwargs["dtype"] = _get_model_dtype(torch)

        load_in_4bit = _env_flag("MODEL_LOAD_IN_4BIT")
        load_in_8bit = _env_flag("MODEL_LOAD_IN_8BIT")
        if load_in_4bit and load_in_8bit:
            raise RuntimeError(
                "MODEL_LOAD_IN_4BIT and MODEL_LOAD_IN_8BIT cannot both be enabled."
            )

        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "bitsandbytes quantization requires a recent transformers install."
                ) from exc

            quantized = True
            model_kwargs["device_map"] = os.getenv("MODEL_DEVICE_MAP", "auto")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type=os.getenv("MODEL_4BIT_QUANT_TYPE", "nf4"),
                bnb_4bit_use_double_quant=_env_flag(
                    "MODEL_4BIT_USE_DOUBLE_QUANT",
                    default=True,
                ),
                bnb_4bit_compute_dtype=_get_model_dtype(torch),
            )

    return model_kwargs, quantized


def _get_model_input_device(model: Any) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            mapped = str(mapped_device)
            if mapped not in {"cpu", "disk", "meta"}:
                return mapped
        if hf_device_map:
            return str(next(iter(hf_device_map.values())))

    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        return str(model_device)

    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return "cpu"
