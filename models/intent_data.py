"""
Data for tf-idf based intent classifier
"""

training_data = [
    # GREETING
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("good afternoon", "greeting"),
    ("hello!", "greeting"),
    ("hi there!", "greeting"),
    ("hola", "greeting"),
    ("hola amigo", "greeting"),
    ("namaste", "greeting"),
    ("yo", "greeting"),
    ("what's up", "greeting"),
    ("how's it going", "greeting"),
    ("hey! how are you", "greeting"),

    # --------------------
    # PRODUCT / PRICING INQUIRY
    # --------------------
    ("what are your plans", "product_inquiry"),
    ("how much does it cost", "product_inquiry"),
    ("tell me about pricing", "product_inquiry"),
    ("what does your product do", "product_inquiry"),
    ("what features do you have", "product_inquiry"),
    ("difference between basic and pro", "product_inquiry"),
    ("what is included in the pro plan", "product_inquiry"),
    ("what are the benefits of your tool", "product_inquiry"),
    ("how is this different from other tools", "product_inquiry"),
    ("do you have a free trial", "product_inquiry"),
    ("can you explain your pricing structure", "product_inquiry"),
    ("what resolutions do you support", "product_inquiry"),
    ("does it generate captions", "product_inquiry"),
    ("is support available on all plans", "product_inquiry"),

    # soft interest / hedged language (STILL inquiry)
    ("i think this might be useful", "product_inquiry"),
    ("this looks interesting", "product_inquiry"),
    ("i am considering using this", "product_inquiry"),
    ("it might be good for my linkedin", "product_inquiry"),
    ("maybe i will use this for youtube", "product_inquiry"),
    ("i am exploring options right now", "product_inquiry"),
    ("Hello, I would like to know more about your product.", "product_inquiry"),
    ("Good morning, what is your pricing?", "product_inquiry"),

    # --------------------
    # HIGH INTENT LEAD
    # --------------------
    ("i want to buy", "high_intent_lead"),
    ("i want to sign up", "high_intent_lead"),
    ("i'm ready to purchase", "high_intent_lead"),
    ("i want the pro plan", "high_intent_lead"),
    ("i want to try for my youtube channel", "high_intent_lead"),
    ("how do i get started right now", "high_intent_lead"),
    ("i'm ready to get started", "high_intent_lead"),
    ("sign me up", "high_intent_lead"),
    ("i want to subscribe", "high_intent_lead"),
    ("help me register", "high_intent_lead"),
    ("i want to create an account", "high_intent_lead"),
    ("i want to upgrade to pro", "high_intent_lead"),
    ("i want to purchase a plan", "high_intent_lead"),
    ("i want to use this for my instagram channel", "high_intent_lead"),
    ("Hello, I would like to sign up for the pro plan.", "high_intent_lead"),
    ("Good evening, how can I get started?", "high_intent_lead"),

    # explicit platform + decision = high intent
    ("i want the pro plan for my youtube", "high_intent_lead"),
    ("i want to join now", "high_intent_lead"),
    ("i want to start today", "high_intent_lead"),
    ("i have decided to go with your product", "high_intent_lead"),
]
