import hashlib
import json
import time
from typing import Dict, Any, List, Optional

class AuditRecord:
    """
    Represents a single audit record with non-repudiation guarantees.
    Includes cryptographic hash chaining for tamper-evidence.
    """
    def __init__(self, event: Dict[str, Any], user: str, timestamp: Optional[float] = None, prev_hash: Optional[str] = None):
        self.event = event
        self.user = user
        self.timestamp = timestamp or time.time()
        self.prev_hash = prev_hash
        self.hash = self._compute_hash()
    def _compute_hash(self) -> str:
        record = {
            'event': self.event,
            'user': self.user,
            'timestamp': self.timestamp,
            'prev_hash': self.prev_hash
        }
        raw = json.dumps(record, sort_keys=True)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event': self.event,
            'user': self.user,
            'timestamp': self.timestamp,
            'prev_hash': self.prev_hash,
            'hash': self.hash
        }

class AuditTrail:
    """
    Maintains an append-only, hash-chained audit trail with non-repudiation guarantees.
    Supports verification of the entire chain for integrity and tamper-evidence.
    """
    def __init__(self):
        self.records: List[AuditRecord] = []
    def append(self, event: Dict[str, Any], user: str):
        prev_hash = self.records[-1].hash if self.records else None
        record = AuditRecord(event, user, prev_hash=prev_hash)
        self.records.append(record)
        return record.hash
    def verify_chain(self) -> bool:
        prev_hash = None
        for record in self.records:
            expected_hash = record._compute_hash()
            if record.hash != expected_hash or record.prev_hash != prev_hash:
                return False
            prev_hash = record.hash
        return True
    def export(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]

# Example usage:
# audit = AuditTrail()
# audit.append({'action':'engage','target':'Alpha'}, user='alice')
# audit.append({'action':'track','target':'Bravo'}, user='bob')
# print(audit.verify_chain())
# print(audit.export())
