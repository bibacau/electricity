from pathlib import Path
import threading


class ToolDenied(Exception):
    pass


class ToolBox:
    """Deliberately small capability set. No shell, unrestricted browser, or Python execution."""
    def __init__(self, root: Path, audit):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stopped = threading.Event()
        self.audit = audit

    def path(self, name):
        if self.stopped.is_set():
            raise ToolDenied("Emergency stop is active")
        p = (self.root / name).resolve()
        if p == self.root or not p.is_relative_to(self.root):
            raise ToolDenied("Files must stay inside the dedicated documents directory")
        return p

    def write(self, name, text):
        p = self.path(name)
        if len(text.encode("utf-8")) > 262144:
            raise ToolDenied("Document exceeds 256 KiB")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: uploads never overwrite an existing document.
        with p.open("x", encoding="utf-8") as f:
            f.write(text)
        self.audit("document.upload", {"path": str(p.relative_to(self.root))})
        return str(p.relative_to(self.root))

    def search(self, query):
        results = []
        for p in sorted(self.root.rglob("*.txt")):
            p = self.path(str(p.relative_to(self.root)))
            if not p.is_file() or p.stat().st_size > 262144:
                continue
            text = p.read_text(encoding="utf-8")
            pos = text.casefold().find(query.casefold())
            if pos >= 0:
                results.append({"name": str(p.relative_to(self.root)), "excerpt": text[max(0,pos-80):pos+240]})
            if len(results) == 20:
                break
        self.audit("document.search", {"matches": len(results)})
        return results
