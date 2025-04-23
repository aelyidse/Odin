import json
import hashlib
import time
from typing import Dict, Any, Optional, List

class TamperEvidentLogEntry:
    """Represents a single log entry with hash chaining for tamper evidence."""
    def __init__(self, data: Dict[str, Any], timestamp: Optional[float] = None, prev_hash: Optional[str] = None):
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.prev_hash = prev_hash or ''
        self.hash = self.compute_hash()
    def compute_hash(self) -> str:
        entry_str = json.dumps({'timestamp': self.timestamp, 'data': self.data, 'prev_hash': self.prev_hash}, sort_keys=True)
        return hashlib.sha256(entry_str.encode('utf-8')).hexdigest()
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'data': self.data,
            'prev_hash': self.prev_hash,
            'hash': self.hash
        }
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TamperEvidentLogEntry':
        entry = cls(d['data'], d['timestamp'], d['prev_hash'])
        assert entry.hash == d['hash'], 'Log entry hash mismatch (possible tampering)'
        return entry

class TamperEvidentLogger:
    """Comprehensive logger with tamper-evident, hash-chained storage."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.entries: List[TamperEvidentLogEntry] = []
        self._load()
    def _load(self):
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    entry = TamperEvidentLogEntry.from_dict(d)
                    self.entries.append(entry)
        except FileNotFoundError:
            pass
    def log(self, data: Dict[str, Any]):
        prev_hash = self.entries[-1].hash if self.entries else ''
        entry = TamperEvidentLogEntry(data, prev_hash=prev_hash)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        self.entries.append(entry)
    def verify(self) -> bool:
        prev_hash = ''
        for entry in self.entries:
            if entry.prev_hash != prev_hash or entry.hash != entry.compute_hash():
                return False
            prev_hash = entry.hash
        return True
    def get_entries(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

# Example usage:
# logger = TamperEvidentLogger('odin_llm_command.log')
# logger.log({'event': 'mission_directive', 'content': ...})
# print('Log valid:', logger.verify())
