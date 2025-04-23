import re
from typing import Dict, Any, List, Optional

class MissionDirective:
    """Structured representation of a parsed mission directive."""
    def __init__(self, action: str, targets: List[str], parameters: Dict[str, Any], raw_text: str):
        self.action = action
        self.targets = targets
        self.parameters = parameters
        self.raw_text = raw_text
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'targets': self.targets,
            'parameters': self.parameters,
            'raw_text': self.raw_text
        }

class MissionNLPParser:
    """
    NLP parser for natural language mission directives, enhanced with specialized military vocabulary and context.
    Recognizes advanced actions, targets, parameters, and context-aware mission semantics.
    Provides grammar validation and ambiguity resolution.
    """
    def __init__(self):
        # Expanded action/target/parameter patterns for military context
        self.action_patterns = [
            (r"(engage|neutralize|track|monitor|illuminate|suppress|jam|escort|intercept|investigate|recon|disrupt|support|evacuate|cover|destroy|defend|attack|disable|divert|observe|survey|acquire|handover|handoff)", 'action'),
        ]
        self.target_patterns = [
            (r"target[s]? ([A-Za-z0-9,\- ]+)", 'targets'),
            (r"hostile[s]? ([A-Za-z0-9,\- ]+)", 'targets'),
            (r"asset[s]? ([A-Za-z0-9,\- ]+)", 'assets'),
            (r"position[s]? ([A-Za-z0-9,\- ]+)", 'positions'),
            (r"objective[s]? ([A-Za-z0-9,\- ]+)", 'objectives'),
            (r"area[s]? ([A-Za-z0-9,\- ]+)", 'areas'),
            (r"grid ([A-Z][0-9]{2,4})", 'grid'),
        ]
        self.parameter_patterns = [
            (r"priority ([A-Za-z0-9]+)", 'priority'),
            (r"duration ([0-9]+) ?(s|sec|seconds|m|min|minutes|h|hr|hours)", 'duration'),
            (r"power ([0-9]+) ?(W|kW)", 'power'),
            (r"ROE ([A-Za-z0-9_\-]+)", 'rules_of_engagement'),
            (r"phase ([A-Za-z0-9_\-]+)", 'mission_phase'),
            (r"window ([0-9:]+)-([0-9:]+)", 'time_window'),
            (r"asset ([A-Za-z0-9_\-]+)", 'asset'),
            (r"effect ([A-Za-z0-9_\-]+)", 'effect'),
            (r"range ([0-9]+) ?(m|km)", 'range'),
            (r"bearing ([0-9]+) ?(deg|degrees)", 'bearing'),
            (r"altitude ([0-9]+) ?(m|ft)", 'altitude'),
        ]
        # Contextual keywords for military domain
        self.context_keywords = [
            'ROE', 'mission phase', 'asset', 'effect', 'window', 'grid', 'objective', 'support', 'evacuate', 'escort', 'divert', 'handover', 'handoff'
        ]

    def parse(self, text: str) -> MissionDirective:
        action = self._extract_action(text)
        targets = self._extract_targets(text)
        parameters = self._extract_parameters(text)
        context = self._extract_context(text)
        # Merge context into parameters for downstream use
        parameters.update(context)
        directive = MissionDirective(action, targets, parameters, text)
        return directive

    def validate(self, directive: MissionDirective) -> Dict[str, Any]:
        """
        Validate the parsed directive for grammar and completeness.
        Returns a dictionary with 'valid', 'missing', and 'ambiguous' fields.
        """
        required_fields = ['action', 'targets']
        missing = []
        ambiguous = []
        # Check for missing required fields
        if not directive.action or directive.action == 'unknown':
            missing.append('action')
        if not directive.targets:
            missing.append('targets')
        # Check for ambiguous or conflicting fields
        if 'priority' in directive.parameters and directive.parameters['priority'] not in ['high', 'medium', 'low', 'urgent']:
            ambiguous.append('priority')
        # Example: ambiguous time window
        if 'window' in directive.parameters:
            window = directive.parameters['window']
            if not window.get('start') or not window.get('end'):
                ambiguous.append('window')
        # Add more ambiguity checks as needed
        valid = not missing and not ambiguous
        return {'valid': valid, 'missing': missing, 'ambiguous': ambiguous}

    def resolve_ambiguity(self, directive: MissionDirective) -> Dict[str, Any]:
        """
        Suggest corrections or request clarification for ambiguous fields.
        Returns a dictionary with suggestions or clarification prompts.
        """
        validation = self.validate(directive)
        suggestions = {}
        for field in validation['missing']:
            if field == 'action':
                suggestions['action'] = 'Specify a clear action (e.g., engage, track, support).'
            if field == 'targets':
                suggestions['targets'] = 'Specify at least one valid target or asset.'
        for field in validation['ambiguous']:
            if field == 'priority':
                suggestions['priority'] = 'Priority should be high, medium, low, or urgent.'
            if field == 'window':
                suggestions['window'] = 'Specify both start and end times for the operational window.'
        return suggestions

    def _extract_action(self, text: str) -> str:
        for pat, _ in self.action_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).lower()
        return "unknown"

    def _extract_targets(self, text: str) -> List[str]:
        found = []
        for pat, name in self.target_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                found += [t.strip() for t in re.split(r",|and|;|/", m.group(1)) if t.strip()]
        return found

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        params = {}
        for pat, name in self.parameter_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                if name == 'duration':
                    value, unit = m.group(1), m.group(2)
                    params['duration'] = {'value': int(value), 'unit': unit}
                elif name == 'power':
                    value, unit = m.group(1), m.group(2)
                    params['power'] = {'value': int(value), 'unit': unit}
                elif name == 'time_window':
                    params['window'] = {'start': m.group(1), 'end': m.group(2)}
                elif name in ['range', 'bearing', 'altitude']:
                    value, unit = m.group(1), m.group(2)
                    params[name] = {'value': int(value), 'unit': unit}
                else:
                    params[name] = m.group(1)
        return params

    def _extract_context(self, text: str) -> Dict[str, Any]:
        context = {}
        for keyword in self.context_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                context[keyword.lower().replace(' ', '_')] = True
        return context

    def parse(self, text: str) -> MissionDirective:
        action = self._extract_action(text)
        targets = self._extract_targets(text)
        parameters = self._extract_parameters(text)
        return MissionDirective(action, targets, parameters, text)

    def _extract_action(self, text: str) -> str:
        for pat, _ in self.action_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).lower()
        return "unknown"

    def _extract_targets(self, text: str) -> List[str]:
        for pat, _ in self.target_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return [t.strip() for t in re.split(r",|and", m.group(1)) if t.strip()]
        return []

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        params = {}
        for pat, name in self.parameter_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                if name == 'duration':
                    value, unit = m.group(1), m.group(2)
                    params['duration'] = {'value': int(value), 'unit': unit}
                elif name == 'power':
                    value, unit = m.group(1), m.group(2)
                    params['power'] = {'value': int(value), 'unit': unit}
                else:
                    params[name] = m.group(1)
        return params

# Example usage:
# parser = MissionNLPParser()
# directive = parser.parse("Engage targets Alpha, Bravo with priority high and duration 30s at power 5kW")
# print(directive.to_dict())
