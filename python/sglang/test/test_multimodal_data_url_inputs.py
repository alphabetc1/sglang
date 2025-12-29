"""
Test multimodal data URL inputs.

Usage:
    python3 -m pytest python/sglang/test/test_multimodal_data_url_inputs.py -v
"""

import base64
import re
from io import BytesIO
from pathlib import Path

import pybase64
import pytest
import requests
from PIL import Image


def _load_common_image_fns():
    """
    Load `ImageData/load_image/get_image_bytes` from `common.py` without importing the
    whole `sglang` package (which pulls in heavy deps like torch/triton).
    """
    repo_root = Path(__file__).resolve().parents[3]
    common_path = repo_root / "python" / "sglang" / "srt" / "utils" / "common.py"
    src = common_path.read_text(encoding="utf-8")

    m_start = re.search(r"(?m)^@dataclass\s*\nclass\s+ImageData\b", src)
    m_end = re.search(r"(?m)^def\s+load_video\b", src)
    assert m_start and m_end and m_start.start() < m_end.start()

    snippet = src[m_start.start() : m_end.start()]
    ns = {
        "base64": base64,
        "BytesIO": BytesIO,
        "Image": Image,
        "pybase64": pybase64,
        "requests": requests,
        "dataclass": __import__("dataclasses").dataclass,
        "Optional": __import__("typing").Optional,
        "Union": __import__("typing").Union,
        "Literal": __import__("typing_extensions").Literal,
        "os": __import__("os"),
    }
    exec(snippet, ns, ns)
    return ns["load_image"], ns["get_image_bytes"]


def _insert_whitespace(s: str, every: int = 60) -> str:
    out = []
    for i, ch in enumerate(s, start=1):
        out.append(ch)
        if i % every == 0:
            out.append("\n \t")
    return "".join(out)


def test_load_image_data_url_allows_base64_with_whitespace():
    load_image, _ = _load_common_image_fns()
    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    b64_ws = _insert_whitespace(b64, every=40)

    data_url = f"data:image/png;base64,{b64_ws}"
    loaded, _ = load_image(data_url)
    assert loaded.size == (10, 10)


def test_load_image_data_url_rejects_pdf_mime_type():
    load_image, _ = _load_common_image_fns()
    pdf_bytes = b"%PDF-1.4\n% minimal\n"
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    data_url = f"data:application/pdf;base64,{_insert_whitespace(b64, every=10)}"

    with pytest.raises(ValueError, match=r"application/pdf not supported"):
        load_image(data_url)


def test_get_image_bytes_data_url_rejects_pdf_mime_type():
    _, get_image_bytes = _load_common_image_fns()
    pdf_bytes = b"%PDF-1.4\n% minimal\n"
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    data_url = f"data:application/pdf;base64,{_insert_whitespace(b64, every=10)}"

    with pytest.raises(ValueError, match=r"application/pdf not supported"):
        get_image_bytes(data_url)
