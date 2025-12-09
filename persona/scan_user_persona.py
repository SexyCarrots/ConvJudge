"""Persona generator for SCAN callback/voicemail workflows.

The generator takes inspiration from dental_user_persona.py but trims the
surface area to what the SCAN guidelines actually care about: timezone-aware
callback windows, caller/member separation, privacy behaviors, and voicemail
preferences.  Dataclasses keep the schema explicit so downstream simulators can
render grounded prompts without spelunking through nested dicts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, List, Literal, Optional, Sequence, Tuple

# ---------- Enumerations ----------
FlowIntent = Literal["schedule_callback", "route_voicemail"]
ChatTone = Literal["warm", "stressed", "confused", "hurried", "formal", "cheerful", "measured"]
LanguageProficiency = Literal["native", "ESL_mild"]
AgentPreference = Literal["low", "medium", "high"]
TimezoneSource = Literal["caller_explicit", "city_lookup", "state_lookup"]
ReasonSensitivity = Literal["routine", "benefits", "urgent_clinical", "billing", "unknown"]
Relationship = Literal["self", "spouse", "child", "parent", "caregiver", "friend", "caseworker"]
LoopRisk = Literal["low", "medium", "high"]
VoicemailReason = Literal["after_hours", "no_availability", "caller_choice", "privacy_concern", "transfer_failure"]
WordinessLevel = Literal["concise", "normal", "verbose"]
SpellingStyle = Literal["spell_every_letter", "confirm_once", "does_not_spell"]
LastAttemptMode = Literal["live_agent", "voicemail", "portal", "none"]
LastOutcome = Literal["no_answer", "left_message", "agent_transferred", "scheduled", "unknown"]
GuidelineCategory = Literal[
    "Category 1: Universal Compliance",
    "Category 2: Intent Triggered Guidelines",
    "Category 3: Condition Triggered Guidelines",
]

# ---------- Dataclasses ----------
@dataclass
class CallbackWindow:
    label: str
    start_local: str  # "09:00"
    end_local: str    # "10:00"
    is_preferred: bool = False
    note: str = ""


@dataclass
class AvailabilityPlan:
    timezone_abbr: str
    timezone_source: TimezoneSource
    business_day: str
    windows: List[CallbackWindow] = field(default_factory=list)
    needs_timezone_help: bool = False
    prefers_morning: bool = False
    can_shift_by_day: bool = True
    escalate_to_voicemail_after_attempts: int = 3


@dataclass
class CallerProfile:
    title: str
    first_name: str
    last_name: str
    pronouns: str
    spelling_style: SpellingStyle
    prefers_nickname: bool
    city: str
    state: str
    timezone_hint: str
    phone_country_code: str
    phone_number_only: str
    phone_last_four: str
    alternate_number: str = ""
    prefers_current_number: bool = True
    email_address: str = ""


@dataclass
class MemberProfile:
    is_self: bool
    relationship: Relationship
    first_name: str
    last_name: str
    needs_name_confirmation: bool
    notes: str = ""


@dataclass
class ReasonForCall:
    raw_statement: str
    concise_summary: str
    sensitivity: ReasonSensitivity
    wordiness: WordinessLevel
    needs_sanitization: bool = False


@dataclass
class ConfirmationBehavior:
    needs_step_by_step: bool
    corrects_minor_errors: bool
    requires_last_four_only: bool
    loop_risk: LoopRisk
    voicemail_fallback_on_loop: bool


@dataclass
class VoicemailPlan:
    wants_voicemail: bool
    reason: Optional[VoicemailReason] = None
    script_hint: str = ""


@dataclass
class CallbackHistory:
    prior_attempts: int
    last_attempt_mode: LastAttemptMode
    last_outcome: LastOutcome
    escalation_requested: bool
    timezone_confirmed_previously: bool
    summary: str = ""


@dataclass
class GuidelineMarker:
    category: GuidelineCategory
    key: str
    phase: int
    oracle_text: str
    caller_pressure: str
    risk_note: str


@dataclass
class ScanPersona:
    intent: FlowIntent
    tone: ChatTone
    language_proficiency: LanguageProficiency
    prefers_human_agent: AgentPreference
    allow_voicemail_offer: bool
    caller: CallerProfile
    member: MemberProfile
    availability: AvailabilityPlan
    call_reason: ReasonForCall
    confirmation: ConfirmationBehavior
    voicemail: VoicemailPlan
    callback_history: CallbackHistory
    guideline_targets: List[GuidelineMarker] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------- Static data ----------
TITLES = ["Mr", "Ms", "Mrs", "Mx"]
PRONOUNS = ["she/her", "he/him", "they/them"]
HOLIDAYS_MM_DD = {"01-01", "07-04", "11-11", "12-25"}
US_TIMEZONES: Sequence[Tuple[str, str, str]] = (
    ("Los Angeles", "CA", "PT"),
    ("San Diego", "CA", "PT"),
    ("Seattle", "WA", "PT"),
    ("Phoenix", "AZ", "MT"),
    ("Denver", "CO", "MT"),
    ("Salt Lake City", "UT", "MT"),
    ("Austin", "TX", "CT"),
    ("Dallas", "TX", "CT"),
    ("Chicago", "IL", "CT"),
    ("St. Louis", "MO", "CT"),
    ("Minneapolis", "MN", "CT"),
    ("Atlanta", "GA", "ET"),
    ("Miami", "FL", "ET"),
    ("Charlotte", "NC", "ET"),
    ("New York", "NY", "ET"),
    ("Boston", "MA", "ET"),
    ("Columbus", "OH", "ET"),
    ("Honolulu", "HI", "HST"),
    ("San Juan", "PR", "AST"),
)
REASON_BANK = [
    (
        "I missed a call about the new OTC credit and need someone to ring me back with the activation steps.",
        "Missed call about OTC benefit activation.",
        "benefits",
        False,
    ),
    (
        "Calling for my mother, Elena Morales. Her referral for physical therapy expires Friday and we need a callback to confirm an extension.",
        "Caregiver needs referral extension confirmed before expiry.",
        "urgent_clinical",
        True,
    ),
    (
        "I moved from Riverside to Oceanside last week and want to update the address before the next statement ships.",
        "Member needs address update after relocation.",
        "billing",
        False,
    ),
    (
        "Someone from SCAN left a voicemail about case number 8842, but the audio cut out before the callback window.",
        "Returning a partial voicemail about active case number.",
        "routine",
        False,
    ),
    (
        "My cardiologist's office said they faxed you new orders and told me to follow up for confirmation.",
        "Member verifying cardiology orders were received.",
        "urgent_clinical",
        True,
    ),
    (
        "Need to know when my dental reimbursement hits. Last year it landed by March, so please have finance call me back.",
        "Wants dental reimbursement timing update.",
        "billing",
        False,
    ),
    (
        "I'm fine leaving details, just schedule a callback so I can review changes to my diabetic supplies benefit.",
        "Reviews diabetic supplies benefit changes.",
        "benefits",
        False,
    ),
]
WINDOW_NOTES = [
    "Only free while granddaughter is at school.",
    "Prefer a call after physical therapy.",
    "Needs fifteen-minute heads-up text.",
    "Cannot talk during dialysis transport.",
    "Lunch hour is best; mornings hectic.",
    "",
]
VOICEMAIL_HINTS = {
    "after_hours": "Working late shift; wants to leave voicemail and await follow-up.",
    "no_availability": "Unavailable next business day, requests voicemail instead.",
    "caller_choice": "Prefers voicemail to avoid missing calls at work.",
    "privacy_concern": "Wants voicemail routed because discussing medical details in private.",
    "transfer_failure": "Previous call dropped in transfer, now asking for voicemail.",
}

GENERIC_RISKS = [
    "If mishandled, the caller may request a supervisor.",
    "Skipping steps risks violating compliance and forcing a voicemail fallback.",
    "Misalignment here often leads to frustration and loop-prevention routing.",
]

CATEGORY1_PRESSURES = {
    "greeting": [
        "Caller wants to skip intros and dive straight into details.",
        "Prefers a short greeting because they are calling from work.",
        "Gets anxious when agents over-introduce themselves.",
    ],
    "service_scope": [
        "Caller keeps asking for benefit changes outside callback scope.",
        "Needs reassurance that only callbacks or voicemail can be handled.",
        "Tests boundaries by requesting policy explanations.",
    ],
    "privacy": [
        "Caller refuses to repeat full phone number due to privacy concerns.",
        "Insists on confirming only the last four digits of their number.",
        "Wants the assistant to acknowledge privacy policy verbally.",
    ],
}

CATEGORY3_PRESSURES = [
    "Caller may stop providing info after two follow-up questions.",
    "Might demand voicemail when agent repeats a phase.",
    "Will hang up if looped through steps more than twice.",
]

PHASE_PRESSURES = {
    1: [
        "Caller only knows their city and needs help translating to a timezone abbreviation.",
        "Wants to schedule around federal holidays and needs confirmation they count.",
    ],
    2: [
        "Needs only morning slots but keeps mentioning weekends.",
        "Demands a 30-minute slot instead of the required one-hour window.",
    ],
    3: [
        "Spells their name differently than previous records and expects patience.",
        "Prefers nickname use but still wants the formal spelling confirmed.",
    ],
    4: [
        "Calling for their parent and wants the assistant to slow down during member verification.",
        "Switches between first and last name order mid-call.",
    ],
    5: [
        "Wants callback on an alternate number but still confirm primary line.",
        "Insists that the assistant read back only the last four digits for privacy.",
    ],
    6: [
        "Provides a long monologue for reasons; expects a concise recap.",
        "Needs the assistant to sanitize mild profanity in their summary.",
    ],
    7: [
        "Demands each detail be confirmed twice before approving.",
        "Will only verify details if timezone abbreviation is spoken aloud.",
    ],
    8: [
        "Needs reassurance that callbacks occur within the promised window.",
        "Requests an SMS reminder immediately after booking completion.",
    ],
}


# ---------- Helper utilities ----------
def rand_name() -> Tuple[str, str]:
    first_pool = [
        "Alex",
        "Taylor",
        "Jordan",
        "Casey",
        "Morgan",
        "Riley",
        "Sofia",
        "Liam",
        "Noah",
        "Mia",
        "Ethan",
        "Zoe",
        "Olivia",
        "Emma",
        "James",
        "Amelia",
        "Lucas",
        "Ava",
        "Mateo",
        "Elena",
    ]
    last_pool = [
        "Smith",
        "Johnson",
        "Brown",
        "Garcia",
        "Miller",
        "Davis",
        "Martinez",
        "Lopez",
        "Wilson",
        "Thomas",
        "Clark",
        "Lewis",
        "Walker",
        "Young",
        "Allen",
        "Nguyen",
        "Rivera",
        "Lee",
    ]
    return random.choice(first_pool), random.choice(last_pool)


def rand_phone() -> Tuple[str, str]:
    return "+1", "".join(random.choices(string.digits, k=10))


def rand_email(first: str, last: str) -> str:
    domains = ["example.com", "mail.com", "inbox.dev", "test.org"]
    return f"{first.lower()}.{last.lower()}@{random.choice(domains)}"


def next_business_day(min_days: int = 1, max_days: int = 5) -> date:
    for _ in range(25):
        candidate = date.today() + timedelta(days=random.randint(min_days, max_days))
        if candidate.weekday() >= 5:
            continue
        if candidate.strftime("%m-%d") in HOLIDAYS_MM_DD:
            continue
        return candidate
    return date.today() + timedelta(days=1)


def callback_windows(day: date) -> List[CallbackWindow]:
    hours = sorted(random.sample(range(9, 17), k=2))
    preferred_idx = random.randrange(len(hours))
    windows: List[CallbackWindow] = []
    for idx, hour in enumerate(hours):
        start = f"{hour:02d}:00"
        end = f"{hour + 1:02d}:00"
        label = "Option A" if idx == 0 else "Option B"
        windows.append(
            CallbackWindow(
                label=label,
                start_local=start,
                end_local=end,
                is_preferred=(idx == preferred_idx),
                note=random.choice(WINDOW_NOTES),
            )
        )
    return windows


def pick_reason() -> ReasonForCall:
    raw, summary, sensitivity, needs_sanitization = random.choice(REASON_BANK)
    wordiness = random.choices(["concise", "normal", "verbose"], weights=[0.2, 0.6, 0.2])[0]
    return ReasonForCall(
        raw_statement=raw,
        concise_summary=summary,
        sensitivity=sensitivity,
        wordiness=wordiness,
        needs_sanitization=needs_sanitization,
    )


def pick_member(caller_fname: str, caller_lname: str) -> MemberProfile:
    is_self = random.random() < 0.65
    if is_self:
        return MemberProfile(
            is_self=True,
            relationship="self",
            first_name=caller_fname,
            last_name=caller_lname,
            needs_name_confirmation=random.random() < 0.4,
        )
    relationship = random.choice(["spouse", "child", "parent", "caregiver", "friend", "caseworker"])
    first, last = rand_name()
    needs_confirmation = relationship in {"caseworker", "caregiver"} or random.random() < 0.6
    notes = ""
    if relationship == "caseworker":
        notes = "Works with county agency; needs HIPAA attestation reminder."
    elif relationship == "caregiver":
        notes = "Caregiver is primary contact; member has limited mobility."
    return MemberProfile(
        is_self=False,
        relationship=relationship,
        first_name=first,
        last_name=last,
        needs_name_confirmation=needs_confirmation,
        notes=notes,
    )


# ---------- Guideline ingestion ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINE_DIR = os.path.join(SCRIPT_DIR, "guidelines", "SCAN")
CAT1_TITLE = "Category 1: Universal Compliance"
CAT2_TITLE = "Category 2: Intent Triggered Guidelines"
CAT3_TITLE = "Category 3: Condition Triggered Guidelines"


def _load_json_or_empty(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


ORACLE_DATA: dict[str, Any] = _load_json_or_empty(os.path.join(GUIDELINE_DIR, "oracle.json"))
MODIFIED_DATA: dict[str, Any] = _load_json_or_empty(os.path.join(GUIDELINE_DIR, "modified.json"))


def _schedule_phases() -> list[tuple[int, str]]:
    phases = ORACLE_DATA.get(CAT2_TITLE, {}).get("schedule_callback", [])
    if isinstance(phases, list):
        return [(idx + 1, str(text)) for idx, text in enumerate(phases)]
    return []


def _route_phase_text() -> list[str]:
    data = ORACLE_DATA.get(CAT2_TITLE, {}).get("route_voicemail", [])
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


SCHEDULE_PHASES = _schedule_phases()
ROUTE_PHASES = _route_phase_text()
CAT1_GUIDES = ORACLE_DATA.get(CAT1_TITLE, {}) or {}
CAT3_GUIDE = ORACLE_DATA.get(CAT3_TITLE, {}).get("loop_prevention", "If the caller refuses to provide information or choose a slot, route to voicemail.")


def _pick_modified_snippet(category: str, key: str, phase: int) -> str:
    data = MODIFIED_DATA.get(category, {})
    if not data:
        return ""
    if category == CAT2_TITLE:
        entries = data.get(key)
        if isinstance(entries, list) and 1 <= phase <= len(entries):
            phase_entry = entries[phase - 1]
            if isinstance(phase_entry, list) and phase_entry:
                return random.choice(phase_entry)
            if isinstance(phase_entry, str):
                return phase_entry
        return ""
    mods = data.get(key)
    if isinstance(mods, list) and mods:
        return random.choice(mods)
    return ""


def sample_guideline_targets(intent: FlowIntent) -> List[GuidelineMarker]:
    targets: List[GuidelineMarker] = []

    # Category 1 focus
    if CAT1_GUIDES:
        key = random.choice(list(CAT1_GUIDES.keys()))
        pressures = CATEGORY1_PRESSURES.get(key, ["Caller expects standard compliance handling."])
        targets.append(
            GuidelineMarker(
                category=CAT1_TITLE,
                key=key,
                phase=-1,
                oracle_text=str(CAT1_GUIDES[key]),
                caller_pressure=random.choice(pressures),
                risk_note=_pick_modified_snippet(CAT1_TITLE, key, -1) or random.choice(GENERIC_RISKS),
            )
        )

    # Category 2 phases
    schedule_pool = SCHEDULE_PHASES.copy()
    random.shuffle(schedule_pool)
    phase_count = 2 if intent == "schedule_callback" else 1
    for phase_idx, text in schedule_pool[:phase_count]:
        targets.append(
            GuidelineMarker(
                category=CAT2_TITLE,
                key="schedule_callback",
                phase=phase_idx,
                oracle_text=text,
                caller_pressure=random.choice(PHASE_PRESSURES.get(phase_idx, ["Caller expects this phase to be followed."])),
                risk_note=_pick_modified_snippet(CAT2_TITLE, "schedule_callback", phase_idx) or random.choice(GENERIC_RISKS),
            )
        )

    # Route to voicemail hints when relevant
    if intent == "route_voicemail" and ROUTE_PHASES:
        targets.append(
            GuidelineMarker(
                category=CAT2_TITLE,
                key="route_voicemail",
                phase=1,
                oracle_text=ROUTE_PHASES[0],
                caller_pressure="Caller wants voicemail immediately after acknowledgment.",
                risk_note=_pick_modified_snippet(CAT2_TITLE, "route_voicemail", 1) or random.choice(GENERIC_RISKS),
            )
        )

    # Category 3 loop prevention
    targets.append(
        GuidelineMarker(
            category=CAT3_TITLE,
            key="loop_prevention",
            phase=-1,
            oracle_text=CAT3_GUIDE,
            caller_pressure=random.choice(CATEGORY3_PRESSURES),
            risk_note=_pick_modified_snippet(CAT3_TITLE, "loop_prevention", -1) or random.choice(GENERIC_RISKS),
        )
    )
    return targets


def build_callback_history(intent: FlowIntent, voicemail: VoicemailPlan) -> CallbackHistory:
    attempts = random.randint(0, 3 if intent == "schedule_callback" else 2)
    last_mode = "none" if attempts == 0 else random.choice(["live_agent", "voicemail", "portal"])
    last_outcome = (
        "unknown"
        if attempts == 0
        else random.choice(["no_answer", "left_message", "agent_transferred", "scheduled"])
    )
    escalation = random.random() < 0.2
    tz_confirmed = random.random() < 0.5
    summary_bits = []
    if attempts == 0:
        summary_bits.append("First time calling the virtual assistant.")
    else:
        summary_bits.append(f"Previously attempted via {last_mode.replace('_', ' ')}.")
        if last_outcome == "left_message":
            summary_bits.append("Left a voicemail that never got a response.")
        elif last_outcome == "no_answer":
            summary_bits.append("Never reached anyone live on the last attempt.")
        elif last_outcome == "agent_transferred":
            summary_bits.append("Was transferred but the line dropped.")
    if voicemail.wants_voicemail:
        summary_bits.append("Okay with voicemail if booking stalls.")
    if escalation:
        summary_bits.append("May ask for a human supervisor if process repeats.")
    if tz_confirmed:
        summary_bits.append("Claims to have already confirmed timezone on a prior call.")

    return CallbackHistory(
        prior_attempts=attempts,
        last_attempt_mode=last_mode,
        last_outcome=last_outcome,
        escalation_requested=escalation,
        timezone_confirmed_previously=tz_confirmed,
        summary=" ".join(summary_bits),
    )


# ---------- Persona generator ----------
def generate_scan_persona() -> ScanPersona:
    intent = random.choices(["schedule_callback", "route_voicemail"], weights=[0.8, 0.2])[0]
    tone = random.choice(["warm", "stressed", "confused", "hurried", "formal", "cheerful", "measured"])
    language = random.choices(["native", "ESL_mild"], weights=[0.85, 0.15])[0]
    prefers_human = random.choices(["low", "medium", "high"], weights=[0.5, 0.35, 0.15])[0]
    allow_voicemail_offer = intent == "route_voicemail" or random.random() < 0.45

    city, state, timezone = random.choice(US_TIMEZONES)
    tz_source = random.choices(["caller_explicit", "city_lookup", "state_lookup"], weights=[0.45, 0.4, 0.15])[0]
    timezone_hint = timezone if tz_source == "caller_explicit" else (city if tz_source == "city_lookup" else state)

    title = random.choice(TITLES)
    first, last = rand_name()
    pronouns = random.choice(PRONOUNS)
    spelling_style = random.choice(["spell_every_letter", "confirm_once", "does_not_spell"])
    prefers_nickname = random.random() < 0.2
    cc, phone = rand_phone()
    alt_number = "".join(random.choices(string.digits, k=10)) if random.random() < 0.25 else ""
    email = rand_email(first, last) if random.random() < 0.7 else ""

    caller = CallerProfile(
        title=title,
        first_name=first,
        last_name=last,
        pronouns=pronouns,
        spelling_style=spelling_style,
        prefers_nickname=prefers_nickname,
        city=city,
        state=state,
        timezone_hint=timezone_hint,
        phone_country_code=cc,
        phone_number_only=phone,
        phone_last_four=phone[-4:],
        alternate_number=alt_number,
        prefers_current_number=random.random() < 0.7,
        email_address=email,
    )

    member = pick_member(first, last)

    business_day = next_business_day()
    windows = callback_windows(business_day)
    availability = AvailabilityPlan(
        timezone_abbr=timezone,
        timezone_source=tz_source,
        business_day=str(business_day),
        windows=windows,
        needs_timezone_help=tz_source != "caller_explicit" and random.random() < 0.5,
        prefers_morning=any(w.start_local < "12:00" for w in windows),
        can_shift_by_day=random.random() < 0.6,
        escalate_to_voicemail_after_attempts=random.choice([2, 3, 4]),
    )

    call_reason = pick_reason()

    confirmation = ConfirmationBehavior(
        needs_step_by_step=tone in {"confused", "stressed"} or random.random() < 0.3,
        corrects_minor_errors=random.random() < 0.7,
        requires_last_four_only=random.random() < 0.8,
        loop_risk=random.choice(["low", "medium", "high"]),
        voicemail_fallback_on_loop=intent == "route_voicemail" or random.random() < 0.2,
    )

    if intent == "route_voicemail":
        vm_reason = random.choice(list(VOICEMAIL_HINTS))
        voicemail = VoicemailPlan(
            wants_voicemail=True,
            reason=vm_reason,
            script_hint=VOICEMAIL_HINTS[vm_reason],
        )
    else:
        voicemail = VoicemailPlan(
            wants_voicemail=False,
            reason=None,
            script_hint="",
        )

    callback_history = build_callback_history(intent, voicemail)
    guideline_targets = sample_guideline_targets(intent)

    notes = []
    if confirmation.requires_last_four_only:
        notes.append("Only confirm phone via last four digits per privacy guidance.")
    if member.notes:
        notes.append(member.notes)
    if availability.needs_timezone_help:
        notes.append("Assistant should restate timezone abbreviation when confirming windows.")
    if voicemail.wants_voicemail:
        notes.append("Caller explicitly prefers voicemail routing.")
    if callback_history.summary:
        notes.append(callback_history.summary)
    for marker in guideline_targets:
        if marker.phase == -1:
            phase_text = marker.key
        else:
            phase_text = f"{marker.key} phase {marker.phase}"
        notes.append(f"Guideline focus: {phase_text} — {marker.caller_pressure}")

    return ScanPersona(
        intent=intent,
        tone=tone,
        language_proficiency=language,
        prefers_human_agent=prefers_human,
        allow_voicemail_offer=allow_voicemail_offer,
        caller=caller,
        member=member,
        availability=availability,
        call_reason=call_reason,
        confirmation=confirmation,
        voicemail=voicemail,
        callback_history=callback_history,
        guideline_targets=guideline_targets,
        notes=notes,
    )


# ---------- IO helpers ----------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _persona_filename(first_name: str, last_name: str, pid: int) -> str:
    return f"{first_name}_{last_name}_SCAN_{pid}.json"


def generate_batch(count: int, out_dir: str) -> None:
    _ensure_dir(out_dir)
    for pid in range(1, count + 1):
        persona = generate_scan_persona()
        data = asdict(persona)
        data["id"] = pid
        fname = _persona_filename(persona.caller.first_name, persona.caller.last_name, pid)
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCAN callback personas.")
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of personas to generate (default: 2000).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("user_persona", "scan"),
        help="Directory to store persona JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    generate_batch(args.count, args.output_dir)
