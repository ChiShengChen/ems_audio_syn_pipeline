"""
EMS Radio Dialogue Prompts for LLM Generation
Based on Virginia Beach / Broadcastify EMS style
"""

SYSTEM_PROMPT = """You are an expert at generating realistic EMS (Emergency Medical Services) radio communications.
Your output will be used to train speech recognition models for EMS dispatch systems.
Generate text that sounds like real radio traffic: brief, sometimes incomplete sentences, 
radio jargon (10-4, copy, roger, standby), and common EMS abbreviations (ALS, BLS, CPR, ETA, etc.).
Use Virginia Beach / Hampton Roads area references when appropriate (street names, landmarks).
Output ONLY the dialogue text, no explanations or metadata."""

# Main prompt for generating EMS dialogues
EMS_DIALOGUE_PROMPT = """Generate a realistic EMS radio communication between a paramedic/dispatcher and another party.
Include the following elements:
- Patient age (or "unknown age")
- Chief complaint (e.g., chest pain, cardiac arrest, MVC, overdose, fall, unconscious)
- Vitals when relevant (conscious/unconscious, breathing/not breathing)
- Location (street address, landmark, or intersection)
- ETA or unit status when appropriate
- Use realistic radio jargon: 10-4, copy, roger, standby, en route, on scene
- Use abbreviations: ALS, BLS, CPR, BP, HR, ETA, PD, tac
- Keep it concise - real radio traffic is brief, sometimes fragmented
- One or two exchanges (dispatcher to unit, or unit to dispatch)
- Do NOT use [x] or placeholders - write complete readable text

Scenario: {scenario}
Chief complaint (if applicable): {chief_complaint}

Generate ONE realistic EMS radio utterance (1-3 sentences):"""

# Batch generation - multiple utterances per call
EMS_BATCH_PROMPT = """Generate {num_utterances} different EMS radio utterances.
Each should be a standalone phrase that could be heard on EMS radio.
Vary: speaker (dispatcher vs paramedic), scenario, chief complaint, location.
Use Virginia Beach area street names when possible.
Format: one utterance per line, no numbering.
Keep each utterance 1-3 sentences, realistic radio style.
Use: 10-4, copy, en route, on scene, ALS, BLS, conscious, breathing, etc."""

# Scenario-specific prompts
SCENARIO_PROMPTS = {
    "dispatch_unit": """Generate a dispatcher assigning a call to an EMS unit.
Include: unit number (e.g., 1623p, rescue 15), address, chief complaint, patient info.
Unit responds with "copy" or "en route".""",
    "patient_report": """Generate a paramedic reporting patient condition to dispatch.
Include: unit ID, patient age/sex, conscious/unconscious, breathing status, chief complaint.
Brief, factual, radio style.""",
    "hospital_notification": """Generate EMS notifying hospital of incoming patient.
Include: unit, ETA, patient condition, chief complaint.
Hospital may acknowledge briefly.""",
    "mass_casualty": """Generate a high-stress EMS radio exchange.
Multiple patients, units coordinating, brief commands.
Use: staging, triage, additional units, ETA.""",
}

# Chief complaint examples for variety
CHIEF_COMPLAINTS = [
    "chest pain", "cardiac arrest", "stroke", "MVC with injuries",
    "overdose", "fall with injury", "unconscious", "breathing difficulty",
    "maternity", "trauma", "gunshot", "gas leak", "structure fire",
    "illness", "choking", "bleed", "headache", "diabetic emergency"
]


def get_prompt(scenario: str = "dispatch_unit", chief_complaint: str = "") -> str:
    """Get formatted prompt for a scenario."""
    scenario_desc = SCENARIO_PROMPTS.get(scenario, EMS_DIALOGUE_PROMPT)
    return EMS_DIALOGUE_PROMPT.format(
        scenario=scenario_desc,
        chief_complaint=chief_complaint or "any"
    )
