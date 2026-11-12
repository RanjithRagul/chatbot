import inspect
import re
import threading
from datetime import datetime
from pathlib import Path


class EventLogger:
    HEADER = "TIME   MODULE   EVENT   INPUT   OUTPUT   ERROR"

    def __init__(self, max_files: int = 20, max_field_length: int = 5000) -> None:
        if max_files < 1 or max_field_length < 1:
            raise ValueError("max_files and max_field_length must be positive")

        self.log_dir = Path(__file__).resolve().parent
        self.project_dir = self.log_dir.parent
        self.max_files = max_files
        self.max_field_length = max_field_length
        self.lock = threading.Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self._create_file()
        self._rotate_files()

    def _create_file(self) -> Path:
        timestamp = datetime.now().strftime("%d %m %y %H %M %S")
        index = 0
        while True:
            suffix = f"_{index:02d}" if index else ""
            path = self.log_dir / f"{timestamp}{suffix}.txt"
            try:
                with path.open("x", encoding="utf-8") as file:
                    file.write(self.HEADER + "\n")
                return path
            except FileExistsError:
                index += 1

    def _rotate_files(self) -> None:
        files = sorted(
            (path for path in self.log_dir.glob("*.txt") if path != self.log_file),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        for path in files[:max(0, len(files) + 1 - self.max_files)]:
            path.unlink()

    def _module_path(self) -> str:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back.f_back
            path = Path(caller.f_code.co_filename).resolve()
            try:
                return path.relative_to(self.project_dir).as_posix()
            except ValueError:
                return path.as_posix()
        finally:
            del frame

    def _field(self, value: object | None) -> str:
        if value is None:
            return ""
        text = str(value).replace("\\", "\\\\")
        text = text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
        text = text.replace("\t", "\\t")
        text = re.sub(r" {3,}", lambda match: "\\x20" * len(match.group()), text)
        text = "".join(char if char.isprintable() else f"\\x{ord(char):02x}" for char in text)
        if len(text) > self.max_field_length:
            text = text[:self.max_field_length] + "...[TRUNCATED]"
        return text

    def log_event(
        self,
        event: str,
        input_data: object | None = None,
        output_data: object | None = None,
        error: object | None = None,
    ) -> None:
        module = self._module_path()
        with self.lock:
            fields = [
                datetime.now().strftime("%H:%M:%S"),
                self._field(module),
                self._field(event),
                self._field(input_data),
                self._field(output_data),
                self._field(error),
            ]
            with self.log_file.open("a", encoding="utf-8") as file:
                file.write("   ".join(fields) + "\n")


logger = EventLogger()
