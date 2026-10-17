"""Complete diagnosis records for offline pipeline fixtures."""


def valid_diagnosis(instance_id="one", **overrides):
    return {
        "instance_id": instance_id,
        "failure_category": "unresolved",
        "exit_status": "Submitted",
        "api_calls": 3,
        "failure_manifestation": "The requested task effect was not verified.",
        "failure_reason": "The agent finished after a tool result without checking the required outcome.",
        "agent_design_issue": "Add an outcome check before the completion decision in agent.py.",
        "affected_component": "verification",
        "severity": "medium",
        "confidence": "medium",
        **overrides,
    }
