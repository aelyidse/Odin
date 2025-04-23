import hashlib
import os
from typing import Optional, Callable

class FirmwareUpdatePackage:
    """Represents a signed firmware/model update package."""
    def __init__(self, filepath: str, signature: str, hash_algo: str = 'sha256'):
        self.filepath = filepath
        self.signature = signature  # Hex-encoded digital signature
        self.hash_algo = hash_algo
        self.file_hash = self.compute_hash()
    def compute_hash(self) -> str:
        h = hashlib.new(self.hash_algo)
        with open(self.filepath, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    def verify_signature(self, verify_fn: Callable[[str, str], bool]) -> bool:
        """Verify the package using a provided signature verification function."""
        return verify_fn(self.file_hash, self.signature)

class SecureUpdater:
    """Handles secure update of firmware and models with verification."""
    def __init__(self, verify_fn: Callable[[str, str], bool], apply_fn: Callable[[str], bool]):
        self.verify_fn = verify_fn
        self.apply_fn = apply_fn
    def update(self, package: FirmwareUpdatePackage) -> bool:
        if not package.verify_signature(self.verify_fn):
            print('Update failed: signature verification failed.')
            return False
        if not self.apply_fn(package.filepath):
            print('Update failed: apply function failed.')
            return False
        print('Update applied successfully.')
        return True

# Example usage (pseudo):
# def verify_signature(hash_val, sig): ... # Use public key crypto
# def apply_firmware(filepath): ... # Flash firmware or load model
# updater = SecureUpdater(verify_signature, apply_firmware)
# pkg = FirmwareUpdatePackage('firmware_v2.bin', signature='...')
# updater.update(pkg)
