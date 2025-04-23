from typing import Dict, Any, List, Optional
import hashlib

class MultiFactorCommandVerifier:
    """
    Implements multi-factor verification for mission and system commands.
    Factors include user authentication, command context, mission state, and cryptographic signatures.
    """
    def __init__(self, authorized_users: Optional[List[str]] = None, require_signature: bool = True, require_context: bool = True):
        self.authorized_users = authorized_users or []
        self.require_signature = require_signature
        self.require_context = require_context

    def verify(self, command: Dict[str, Any], user: str, context: Dict[str, Any], signature: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifies command authenticity and appropriateness using multiple factors.
        Returns dict with 'verified', 'failures', and 'details'.
        """
        failures = []
        details = {}
        # 1. User authentication
        if self.authorized_users and user not in self.authorized_users:
            failures.append('user_auth')
            details['user_auth'] = f"User '{user}' not authorized."
        # 2. Command context (e.g., mission phase, ROE)
        if self.require_context:
            if 'mission_phase' not in context or 'roe' not in context:
                failures.append('context')
                details['context'] = 'Missing mission_phase or ROE.'
        # 3. Cryptographic signature (simple hash-based demo)
        if self.require_signature:
            expected_sig = self._compute_signature(command, user)
            if not signature or signature != expected_sig:
                failures.append('signature')
                details['signature'] = 'Invalid or missing signature.'
        # 4. Command grammar/ambiguity check (optional)
        if 'action' not in command or command['action'] == 'unknown':
            failures.append('grammar')
            details['grammar'] = 'Command action missing or ambiguous.'
        verified = not failures
        return {'verified': verified, 'failures': failures, 'details': details}

    def _compute_signature(self, command: Dict[str, Any], user: str) -> str:
        """
        Computes a simple hash signature for demo purposes.
        """
        raw = str(command) + user
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]  # Shortened for demo

# Example usage:
# verifier = MultiFactorCommandVerifier(authorized_users=['alice','bob'])
# command = {'action': 'engage', 'parameters': {'priority': 'high'}}
# context = {'mission_phase': 'engagement', 'roe': 'tight'}
# sig = verifier._compute_signature(command, 'alice')
# result = verifier.verify(command, 'alice', context, sig)
# print(result)
