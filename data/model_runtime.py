from __future__ import annotations

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

            model_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.device and self.device.startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
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

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TRAIN_SAMPLE_TOKENS,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

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

            model_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.device and self.device.startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.float16

            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
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

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

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
