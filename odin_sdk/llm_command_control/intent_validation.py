from typing import Dict, Any, List, Optional

class IntentValidator:
    """
    Validates parsed command or mission intent against mission parameters, ROE, and operational constraints.
    Flags intent that violates or conflicts with mission objectives, rules, or boundaries.
    """
    def __init__(self, mission_parameters: Dict[str, Any]):
        self.mission_parameters = mission_parameters

    def validate_intent(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the intent (directive) is valid within the current mission parameters.
        Returns dict with 'valid', 'violations', and 'details'.
        """
        violations = []
        details = {}
        # 1. Check action against allowed actions
        allowed_actions = self.mission_parameters.get('allowed_actions', [])
        if allowed_actions and directive.get('action') not in allowed_actions:
            violations.append('action')
            details['action'] = f"Action '{directive.get('action')}' not permitted."
        # 2. Check targets against protected or restricted targets
        protected_targets = self.mission_parameters.get('protected_targets', [])
        for t in directive.get('targets', []):
            if t in protected_targets:
                violations.append('protected_target')
                details.setdefault('protected_target', []).append(f"Target '{t}' is protected.")
        # 3. Check ROE compliance
        roe = self.mission_parameters.get('roe')
        if roe and directive.get('parameters', {}).get('rules_of_engagement') not in [None, roe]:
            violations.append('roe')
            details['roe'] = f"Directive ROE '{directive.get('parameters', {}).get('rules_of_engagement')}' does not match mission ROE '{roe}'."
        # 4. Check operational boundaries (e.g., area, time window)
        op_area = self.mission_parameters.get('operational_area')
        if op_area and 'area' in directive.get('parameters', {}):
            if directive['parameters']['area'] not in op_area:
                violations.append('operational_area')
                details['operational_area'] = f"Area '{directive['parameters']['area']}' not in mission area."
        time_window = self.mission_parameters.get('time_window')
        if time_window and 'window' in directive.get('parameters', {}):
            win = directive['parameters']['window']
            if not (time_window['start'] <= win['start'] <= time_window['end'] and time_window['start'] <= win['end'] <= time_window['end']):
                violations.append('time_window')
                details['time_window'] = f"Directive window {win} outside mission window {time_window}."
        # 5. Custom mission constraints
        for k, v in self.mission_parameters.get('custom_constraints', {}).items():
            if k in directive.get('parameters', {}) and directive['parameters'][k] != v:
                violations.append(f'constraint_{k}')
                details[f'constraint_{k}'] = f"Parameter '{k}' value '{directive['parameters'][k]}' does not match required '{v}'."
        valid = not violations
        return {'valid': valid, 'violations': violations, 'details': details}

# Example usage:
# mission_params = {'allowed_actions': ['engage','track'], 'roe': 'tight', 'protected_targets': ['Alpha'], 'operational_area': ['Zone1','Zone2'], 'time_window': {'start': '08:00', 'end': '18:00'}}
# validator = IntentValidator(mission_params)
# directive = {'action': 'engage', 'targets': ['Alpha', 'Bravo'], 'parameters': {'rules_of_engagement': 'tight', 'area': 'Zone3', 'window': {'start': '07:00', 'end': '09:00'}}}
# print(validator.validate_intent(directive))
