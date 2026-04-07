from __future__ import annotations

import json
import logging
import uuid
from typing import Iterable, List, Optional

from fastapi import Request
from fastapi.responses import ORJSONResponse
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
)

from sglang.srt.entrypoints.compat.base import (
    CompatAdapter,
    CompatRouteSpec,
    CompatServiceRegistry,
)
from sglang.srt.entrypoints.openai.protocol import V1RerankReqInput
from sglang.srt.entrypoints.openai.serving_rerank import _is_qwen3_reranker_template

logger = logging.getLogger(__name__)


class BailianQwenRerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    instruct: Optional[str] = None
    return_documents: bool = True

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v):
        if v is not None and v < 1:
            raise ValueError("Value error, parameter top_n should be larger than 0.")
        return v

    def to_v1_rerank_request(self) -> V1RerankReqInput:
        return V1RerankReqInput(
            query=self.query,
            documents=self.documents,
            instruct=self.instruct,
            top_n=self.top_n,
            return_documents=self.return_documents,
        )


class BailianRerankDocument(BaseModel):
    text: str


class BailianRerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[BailianRerankDocument] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.document is None:
            data.pop("document", None)
        return data


class BailianRerankOutput(BaseModel):
    results: List[BailianRerankResult]


class BailianRerankUsage(BaseModel):
    total_tokens: int = Field(default=0)


class BailianRerankResponse(BaseModel):
    output: BailianRerankOutput
    usage: BailianRerankUsage
    request_id: str


class BailianErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str


class BailianQwen3RerankAdapter(CompatAdapter):
    """Bailian-compatible adapter for qwen3-rerank."""

    name = "bailian_qwen3_rerank"

    def route_specs(self) -> Iterable[CompatRouteSpec]:
        return [
            CompatRouteSpec(path="/reranks", methods=("POST",), include_in_schema=True),
            CompatRouteSpec(
                path="/v1/reranks", methods=("POST",), include_in_schema=True
            ),
            CompatRouteSpec(
                path="/compatible-api/v1/reranks",
                methods=("POST",),
                include_in_schema=False,
            ),
        ]

    async def handle_request(
        self,
        raw_request: Request,
        services: CompatServiceRegistry,
    ) -> ORJSONResponse:
        request_id = str(uuid.uuid4())
        rerank_service = services.rerank
        if rerank_service is None:
            return self._error_response(
                code="InternalError",
                message="Rerank service is not available.",
                request_id=request_id,
                status_code=500,
            )

        media_type = (
            raw_request.headers.get("content-type", "").lower().split(";", 1)[0]
        )
        if media_type != "application/json":
            return self._error_response(
                code="InvalidParameter",
                message="Unsupported Media Type: Only 'application/json' is allowed",
                request_id=request_id,
                status_code=400,
            )

        try:
            payload = await raw_request.json()
        except json.JSONDecodeError as e:
            return self._error_response(
                code="InvalidParameter",
                message=f"Invalid JSON body: {e.msg}",
                request_id=request_id,
                status_code=400,
            )

        try:
            request = BailianQwenRerankRequest(**payload)
        except ValidationError as e:
            return self._error_response(
                code="InvalidParameter",
                message=str(e),
                request_id=request_id,
                status_code=400,
            )

        if request.model != "qwen3-rerank":
            return self._error_response(
                code="InvalidParameter",
                message="Only model 'qwen3-rerank' is supported on this endpoint.",
                request_id=request_id,
                status_code=400,
            )

        v1_request = request.to_v1_rerank_request()
        error_msg = rerank_service._validate_request(v1_request)
        if error_msg:
            return self._error_response(
                code="InvalidParameter",
                message=error_msg,
                request_id=request_id,
                status_code=400,
            )

        chat_template = getattr(
            rerank_service.tokenizer_manager.tokenizer,
            "chat_template",
            None,
        )
        if not _is_qwen3_reranker_template(
            chat_template if isinstance(chat_template, str) else ""
        ):
            return self._error_response(
                code="InvalidParameter",
                message=(
                    "The current server is not configured as a qwen3-rerank endpoint. "
                    "Please launch Qwen3-Reranker with the qwen3 reranker chat template."
                ),
                request_id=request_id,
                status_code=400,
            )

        try:
            scores, prompt_tokens = await rerank_service._score_text_reranker_request(
                request=v1_request,
                raw_request=raw_request,
                chat_template=chat_template,
            )
        except ValueError as e:
            return self._error_response(
                code="InvalidParameter",
                message=str(e),
                request_id=request_id,
                status_code=400,
            )
        except Exception as e:
            logger.exception("Error handling Bailian rerank request")
            return self._error_response(
                code="InternalError",
                message=str(e),
                request_id=request_id,
                status_code=500,
            )

        rerank_results = rerank_service._build_rerank_response(scores, v1_request)
        response = BailianRerankResponse(
            output=BailianRerankOutput(
                results=[
                    BailianRerankResult(
                        index=result.index,
                        relevance_score=result.score,
                        document=(
                            BailianRerankDocument(text=result.document)
                            if result.document is not None
                            else None
                        ),
                    )
                    for result in rerank_results
                ]
            ),
            usage=BailianRerankUsage(total_tokens=prompt_tokens),
            request_id=request_id,
        )
        return ORJSONResponse(content=response.model_dump())

    def _error_response(
        self,
        *,
        code: str,
        message: str,
        request_id: str,
        status_code: int,
    ) -> ORJSONResponse:
        error = BailianErrorResponse(
            code=code,
            message=message,
            request_id=request_id,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)
