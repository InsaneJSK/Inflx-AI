"""
State manager for AutoStream assistant.
Tracks conversation context including chat history, detected intents,
and lead capture details.
"""

from typing import List, Optional, Dict

class ConversationState:
    """
    Tracks conversation context for the AutoStream assistant.
    
    Stored:
    - chat history
    - last detected intent
    - whether we are collecting lead details
    - lead fields: name, email, platform
    """

    MAX_TURNS = 5

    def __init__(self):
        # memory
        self.history: List[Dict[str, str]] = []
        self.last_intent: Optional[str] = None

        self.collecting_lead: bool = False
        self.name: Optional[str] = None
        self.email: Optional[str] = None
        self.platform: Optional[str] = None

    def add_turn(self, role, message: str):
        """
        Adds a messages to the history
        
        :param role: "User" or "Assistant"
        :param message: the message content
        :type message: str
        """
        self.history.append({"role": role, "content": message})
        if len(self.history) > self.MAX_TURNS:
            self.history = self.history[-self.MAX_TURNS:]

    def missing_lead_fields(self):
        """
        Returns a list of missing lead fields.
        :return: list of missing fields
        :rtype: List[str]
        """
        missing = []
        if not self.name:
            missing.append("name")
        if not self.email:
            missing.append("email")
        if not self.platform:
            missing.append("platform")
        return missing

    def reset_lead_capture(self):
        """Resets lead capture state."""
        self.collecting_lead = False
        self.name = None
        self.email = None
        self.platform = None

if __name__ == "__main__":
    s = ConversationState()
    s.add_turn("User", "I want to sign up")
    s.collecting_lead = True
    s.add_turn("Assistant", "Sure! Can I have your name?")
    s.name = "Jaspreet"
    s.add_turn("User", "Jaspreet")
    print("History:", s.history)
    print("Missing:", s.missing_lead_fields())
    # Test exceeding max turns
    for i in range(8):
        s.add_turn("user", f"message {i}")

    print(s.history)  # should only show last 5
