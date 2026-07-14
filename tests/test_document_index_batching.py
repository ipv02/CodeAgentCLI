from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code_agent_cli.document_index import DocumentIndexService


class FakeBatchIndexService(DocumentIndexService):
    def __init__(self, root: Path) -> None:
        super().__init__(
            db_path=root / "index.db",
            report_path=root / "report.json",
        )
        self.batch_sizes: list[int] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(index + 1), 0.5] for index, _text in enumerate(texts)]


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class DocumentIndexBatchingTests(unittest.TestCase):
    def test_selected_index_uses_embedding_batches(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        selected_paths = []
        for index in range(5):
            path = root / f"doc-{index}.md"
            path.write_text(f"# Document {index}\n\nText {index}.\n", encoding="utf-8")
            selected_paths.append(path.name)
        service = FakeBatchIndexService(root)

        report = service.index_path(
            str(root),
            strategies=["structural"],
            selected_paths=selected_paths,
            embedding_batch_size=2,
        )

        self.assertEqual(service.batch_sizes, [2, 2, 1])
        self.assertEqual(report["documents"]["count"], 5)
        self.assertEqual(report["embedding"]["batch_size"], 2)

    def test_embed_batch_uses_modern_ollama_endpoint(self) -> None:
        service = DocumentIndexService()
        response = FakeHTTPResponse({"embeddings": [[1, 2], [3, 4]]})

        with patch("code_agent_cli.document_index.urlopen", return_value=response) as urlopen:
            embeddings = service.embed_batch(["first", "second"])

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/embed")
        self.assertEqual(payload["input"], ["first", "second"])
        self.assertEqual(embeddings, [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
