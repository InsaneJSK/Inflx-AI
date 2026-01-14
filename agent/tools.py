"""
Tools for Inflx AI agent.

Currently includes:
- mock lead capture tool
"""

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates sending captured lead details to CRM/webhook.
    For assignment demo: we only print + return string.
    Args:
        name: Lead's name
        email: Lead's email
        platform: Platform of interest
    Returns:
        Confirmation string
    """
    if not (name and email and platform):
        raise ValueError("All fields (name, email, platform) must be provided")
    result = {
    "status": "success",
    "name": name,
    "email": email,
    "platform": platform
    }
    print(result)
    return result

if __name__ == "__main__":
    from agent.state_manager import ConversationState

    s = ConversationState()
    s.collecting_lead = True
    s.name = "Jaspreet"
    s.email = "jaspreet@example.com"
    s.platform = "WhatsApp"

    out = mock_lead_capture(s.name, s.email, s.platform)

    print("Tool returned:", out)
    print("Collecting lead mode:", s.collecting_lead)
    print("Missing fields now:", s.missing_lead_fields())
