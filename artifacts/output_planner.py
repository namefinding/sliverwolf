from __future__ import annotations

from pathlib import Path


class OutputArtifactPlanner:
    @staticmethod
    def build_write_arguments(*, output_file: str, content: str, delivery_intent) -> tuple[str, dict]:
        suffix = Path(output_file).suffix.lower()
        title = None if delivery_intent is None else getattr(delivery_intent, "title", None)
        if suffix == ".docx":
            return "file.write_docx", {"path": output_file, "title": title, "content": content, "overwrite": True}
        if suffix == ".xlsx":
            return "file.write_xlsx", {"path": output_file, "title": title, "content": content, "overwrite": True}
        if suffix == ".pptx":
            return "file.write_pptx", {"path": output_file, "title": title, "content": content, "overwrite": True}
        return "file.write", {"path": output_file, "content": content, "overwrite": True}
