
import random
import string
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import List, Literal
import os
import json

# Human-friendly intents (what the caller wants to do)
FlowIntent = Literal[
    "schedule_consultation",         # wants to book a consultation
    "check_eligibility",             # wants to see if they qualify (hard checks)
    "ask_insurance_and_financing",   # wants insurance transparency and HCF options
    "learn_treatment_options",       # wants general consult/treatment info
]
ChatTone = Literal["hopeful", "anxious", "curious", "impatient", "polite", "decisive"]
LanguageProficiency = Literal["fluent", "ESL_mild"]
AgentPreference = Literal["low", "medium", "high"]
InsuranceType = Literal["none", "dental", "health", "both"]
CreditBand = Literal["650_or_higher", "below_650", "unknown"]
NotifChannel = Literal["email", "sms", "both"]
SedationPref = Literal["yes", "no", "unsure"]
HandoffTrigger = Literal[
    "",  # no special trigger
    "refuses_dental_history",
    "scan_copy_cost",
    "legal_mail_address",
    "attorney_office_caller",
    "accessibility_blind_deaf",
    "existing_implant_repair",
    "traveling_implant_damage",
    "verify_appointment_datetime",
    "cannot_reach_center",
    "no_confirmation_call",
    "asl_interpreter_virtual",
    "patient_deceased_dnc",
    "hospital_or_dentist_office",
    "double_consult_request",
    "did_not_schedule",
    "running_late",
    "fraudulent_loan",
    "third_party_confirm_appt",
    "second_or_repeat_consult",
    "mend_troubleshooting",
    "online_or_virtual_request",
    "needs_prescription",
    "refund_request",
    "pregnancy",
    "outbound_number_spam",
    "under_18_dob",
]


# ---------- Helpers ----------
TITLES = ["Mr", "Ms", "Mx"]
GENDERS = ["M", "F", "X"]

US_CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"), ("Phoenix", "AZ"),
    ("Philadelphia", "PA"), ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"), ("San Jose", "CA"),
    ("Austin", "TX"), ("Jacksonville", "FL"), ("Fort Worth", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"),
    ("San Francisco", "CA"), ("Indianapolis", "IN"), ("Seattle", "WA"), ("Denver", "CO"), ("Washington", "DC")
]

STREETS = ["Oak St","Maple Ave","Pine Rd","Cedar Ln","Elm St","Sunset Blvd","Broadway","2nd St","Lakeview Dr"]

def rand_name():
    first_pool = ["Alex","Taylor","Jordan","Casey","Morgan","Avery","Riley","Sofia","Liam","Noah",
                  "Mia","Ethan","Aiden","Zoe","Olivia","Emma","James","Amelia","Lucas","Ava"]
    last_pool  = ["Smith","Johnson","Brown","Garcia","Miller","Davis","Martinez","Lopez","Wilson","Thomas",
                  "Clark","Lewis","Walker","Young","Allen"]
    return random.choice(first_pool), random.choice(last_pool)

def rand_phone():
    return "+1", "".join(random.choices(string.digits, k=10))

def rand_email(first, last):
    domains = ["example.com","mail.com","inbox.dev","test.org"]
    return f"{first.lower()}.{last.lower()}@{random.choice(domains)}"

def future_date(days_min=3, days_max=30):
    return date.today() + timedelta(days=random.randint(days_min, days_max))

def random_time_window():
    return random.choice(["early morning","morning","midday","afternoon","evening"])

def rand_street_addr():
    return f"{random.randint(100, 9999)} {random.choice(STREETS)}"

def rand_zip():
    return f"{random.randint(10000, 99999)}"

# ---------- Data classes (existing) ----------
@dataclass
class ContactAddress:
    street: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""

@dataclass
class DentalCondition:
    missing_broken_failing: bool = False
    gum_disease: bool = False
    current_solutions: List[str] = field(default_factory=list)
    pain_present: bool = False
    pain_level_1_to_5: int = 0
    eating_issues: bool = False
    foods_avoided: List[str] = field(default_factory=list)

@dataclass
class InsuranceFinancing:
    has_insurance: bool = False
    insurance_type: InsuranceType = "none"
    insurance_provider: str = ""
    credit_score_band: CreditBand = "unknown"
    has_cosigner_available: bool = False
    interested_in_financing: bool = False
    insurance_assurance_eligible: bool = False
    denture_trade_in_eligible: bool = False

@dataclass
class AppointmentPrefs:
    preferred_center_city: str = ""
    preferred_center_state: str = ""
    preferred_center_zip: str = ""
    preferred_date: str = ""
    preferred_time_window: str = ""
    sedation_preference: SedationPref = "unsure"
    same_day_teeth_interest: bool = False
    will_accept_earliest_slot: bool = True

@dataclass
class HandoffContext:
    trigger: HandoffTrigger = ""
    third_party_role: str = ""            # "attorney"|"hospital"|"transport_or_insurance"|""
    accessibility_need: str = ""          # "asl"|"blind"|"deaf"|""
    pregnancy: bool = False
    under_18: bool = False
    is_existing_patient: bool = False
    has_existing_appointment: bool = False
    running_late: bool = False
    wants_virtual_only: bool = False
    needs_prescription: bool = False
    refund_request: bool = False
    finance_issue: str = ""               # "fraudulent_loan"|""
    info_request: str = ""                # "scan_copy_cost"|"legal_mail_address"|""
    cannot_reach_center: bool = False
    traveling_implant_damage: bool = False
    verify_appointment: bool = False
    did_not_schedule: bool = False
    second_or_repeat_consult: bool = False
    existing_implant_repair: bool = False
    double_consult_request: bool = False
    outbound_number_spam: bool = False
    asl_for_virtual: bool = False

# ---------- NEW: Rich personality / life context ----------
RiskLevel = Literal["low","medium","high"]
TriLevel = Literal["low","medium","high"]

@dataclass
class PersonalityProfile:
    # Balanced Big Five (1-5) + explicit positives/negatives
    openness: int = 3
    conscientiousness: int = 3
    extraversion: int = 3
    agreeableness: int = 3
    neuroticism: int = 3
    positive_traits: List[str] = field(default_factory=list)
    negative_traits: List[str] = field(default_factory=list)
    coping_style: Literal["problem_focused","emotion_focused","avoidant"] = "problem_focused"
    conflict_style: Literal["accommodating","competing","compromising","collaborating","avoiding"] = "compromising"
    decision_style: Literal["deliberative","impulsive","delegates","seeks_reassurance"] = "deliberative"
    trust_sales: TriLevel = "medium"
    trust_clinical: TriLevel = "medium"
    skepticism_triggers: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    dealbreakers: List[str] = field(default_factory=list)
    frustration_threshold: TriLevel = "medium"
    procrastination_tendency: TriLevel = "medium"

@dataclass
class LifeContext:
    employment_status: Literal["full_time","part_time","unemployed","retired","student"] = "full_time"
    work_schedule_constraint: Literal["fixed_shifts","flexible","gig","unemployed"] = "flexible"
    caregiver_responsibility: Literal["none","children","elderly","both"] = "none"
    transport_access: Literal["car","public_transport","rideshare_only","none"] = "car"
    distance_to_center_minutes: int = 25
    budget_sensitivity: TriLevel = "medium"
    budget_cap_usd: int = 5000
    previous_negative_experiences: List[str] = field(default_factory=list)
    barriers: List[str] = field(default_factory=list)  # e.g., "fear_of_dentist","time_off","transport","language"

@dataclass
class ComplianceBehavior:
    adherence_probability: int = 70        # 0-100
    cancellation_probability: int = 15     # 0-100
    no_show_risk: RiskLevel = "medium"
    price_sensitivity: TriLevel = "medium"
    financing_comfort: TriLevel = "medium"
    pain_tolerance: TriLevel = "medium"
    skepticism_level: TriLevel = "medium"  # general skepticism (overlay with personality)
    follow_up_responsiveness: TriLevel = "medium"

# ---------- Persona container ----------
@dataclass
class DentalPersona:
    # Flow and conversation behavior
    intent: FlowIntent
    tone: ChatTone
    language_proficiency: LanguageProficiency
    prefers_human_agent: AgentPreference
    allow_human_anytime: bool = True
    in_a_hurry: bool = False
    filler_tolerance: TriLevel = "medium"

    # Identity
    title: str = ""
    first_name: str = ""
    last_name: str = ""
    gender: str = ""
    date_of_birth: str = ""
    dob_provided: bool = True

    # Contact
    phone_country_code: str = ""
    phone_number_only: str = ""
    email_address: str = ""
    preferred_notification_channel: NotifChannel = "both"
    address: ContactAddress = field(default_factory=ContactAddress)

    # Condition + appointment + financing
    condition: DentalCondition = field(default_factory=DentalCondition)
    appointment: AppointmentPrefs = field(default_factory=AppointmentPrefs)
    insurance_financing: InsuranceFinancing = field(default_factory=InsuranceFinancing)
    # Category 3: Human Agent Handoff scenario (to exercise transfer paths)
    handoff: HandoffContext = field(default_factory=HandoffContext)

    # NEW sections
    personality: PersonalityProfile = field(default_factory=PersonalityProfile)
    life_context: LifeContext = field(default_factory=LifeContext)
    compliance: ComplianceBehavior = field(default_factory=ComplianceBehavior)

    # Extras
    accompaniment_planned: bool = False
    accompaniment_name: str = ""
    privacy_policy_ack: bool = True

# ---------- Samplers ----------
def _pick_identity_flat() -> dict:
    fn, ln = rand_name()
    refuse = random.random() < 0.05
    if refuse:
        dob, provided = "2002-01-01", False
    else:
        dob_year = random.randint(1955, 2002)
        dob = f"{dob_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        provided = True
    return {
        "title": random.choice(TITLES),
        "first_name": fn,
        "last_name": ln,
        "gender": random.choice(GENDERS),
        "date_of_birth": dob,
        "dob_provided": provided,
    }

def _pick_condition(for_consult: bool) -> DentalCondition:
    if for_consult:
        missing_broken = True
        gum = random.random() < 0.4
    else:
        missing_broken = random.random() < 0.6
        gum = random.random() < 0.3
    current_solutions_pool = ["denture","partial denture","bridge","crown","dental implants"]
    have_solution = random.random() < (0.55 if for_consult else 0.4)
    current_solutions = random.sample(current_solutions_pool, k=random.randint(1,2)) if have_solution else []
    pain_present = random.random() < (0.65 if for_consult else 0.4)
    pain_level = random.choices([1,2,3,4,5], weights=[0.1,0.2,0.35,0.25,0.1])[0] if pain_present else 0
    eating_issues = random.random() < (0.55 if for_consult else 0.35)
    foods_pool = ["steak","nuts","apples","corn on the cob","crusty bread","carrots"]
    foods_avoided = random.sample(foods_pool, k=random.randint(1,2)) if eating_issues else []
    return DentalCondition(missing_broken_failing=missing_broken, gum_disease=gum,
                           current_solutions=current_solutions, pain_present=pain_present,
                           pain_level_1_to_5=pain_level, eating_issues=eating_issues,
                           foods_avoided=foods_avoided)

def _pick_financing(condition: DentalCondition) -> InsuranceFinancing:
    has_ins = random.random() < 0.6
    ins_type = random.choices(["dental","health","both","none"], weights=[0.45,0.15,0.05,0.35])[0]
    if not has_ins:
        ins_type = "none"
    provider = "" if ins_type == "none" else random.choice(
        ["Delta Dental","MetLife","Aetna","Cigna","UnitedHealthcare","Blue Cross","Guardian"]
    )
    credit = random.choices(["650_or_higher","below_650","unknown"], weights=[0.5,0.3,0.2])[0]
    interested = random.random() < 0.7
    insurance_assurance_eligible = ins_type != "none"
    denture_trade_in_eligible = ("denture" in condition.current_solutions or "partial denture" in condition.current_solutions) and (ins_type == "none")
    cosigner = credit == "below_650" and (random.random() < 0.5)
    return InsuranceFinancing(has_insurance=has_ins, insurance_type=ins_type, insurance_provider=provider,
                              credit_score_band=credit, has_cosigner_available=bool(cosigner),
                              interested_in_financing=interested, insurance_assurance_eligible=insurance_assurance_eligible,
                              denture_trade_in_eligible=denture_trade_in_eligible)

def _pick_appointment(in_a_hurry: bool) -> AppointmentPrefs:
    city, state = random.choice(US_CITIES)
    days_min, days_max = (1,7) if in_a_hurry else (3,21)
    d = future_date(days_min=days_min, days_max=days_max)
    tw = random_time_window()
    sed_pref = random.choices(["yes","no","unsure"], weights=[0.4,0.1,0.5])[0]
    same_day = random.random() < 0.35
    earliest_ok = random.random() < (0.75 if in_a_hurry else 0.5)
    return AppointmentPrefs(preferred_center_city=city, preferred_center_state=state, preferred_center_zip=rand_zip(),
                            preferred_date=str(d), preferred_time_window=tw,
                            sedation_preference=sed_pref, same_day_teeth_interest=same_day,
                            will_accept_earliest_slot=earliest_ok)

def _pick_address() -> ContactAddress:
    city, state = random.choice(US_CITIES)
    return ContactAddress(street=rand_street_addr(), city=city, state=state, zip_code=rand_zip())

# NEW: samplers for personality & context
POS_TRAITS = ["resilient","optimistic","thorough","punctual","cooperative","curious","disciplined","empathetic","decisive","calm"]
NEG_TRAITS = ["skeptical","stubborn","anxious","impatient","indecisive","avoidant","distrustful","perfectionistic","impulsive","defensive"]
MOTIVATIONS = ["eat_comfortably","restore_confidence","avoid_future_pain","look_younger","speak_clearly","special_event","doctor_recommendation","family_encouragement"]
FEARS = ["pain","cost","surgery","failure_of_implants","time_off_work","judgment","downtime","anesthesia"]
DEALBREAKERS = ["no_upfront_costs","no_long_waits","no_pressure_sales","transparent_pricing","only_weekend_appointments"]
SKEPTIC_TRIGGERS = ["pushy_sales","hidden_fees","rushed_explanations","conflicting_information","overuse_of_jargon"]
BARRIERS = ["fear_of_dentist","childcare","eldercare","transport","time_off","language","work_shift","past_bad_experience"]
NEG_EXPERIENCES = ["painful_previous_procedure","billing_dispute","missed_followup_call","feel_talked_down_to","financing_denied"]

def _pick_personality() -> PersonalityProfile:
    # Balanced but varied Big Five
    bigfive = {k: random.randint(1,5) for k in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]}
    pos = random.sample(POS_TRAITS, k=random.randint(2,4))
    neg = random.sample(NEG_TRAITS, k=random.randint(2,4))
    coping = random.choice(["problem_focused","emotion_focused","avoidant"])
    conflict = random.choice(["accommodating","competing","compromising","collaborating","avoiding"])
    decision = random.choice(["deliberative","impulsive","delegates","seeks_reassurance"])
    trust_sales = random.choices(["low","medium","high"], weights=[0.45,0.4,0.15])[0]
    trust_clinical = random.choices(["low","medium","high"], weights=[0.2,0.5,0.3])[0]
    skept_trig = random.sample(SKEPTIC_TRIGGERS, k=random.randint(1,3))
    motivations = random.sample(MOTIVATIONS, k=random.randint(2,4))
    fears = random.sample(FEARS, k=random.randint(2,4))
    dealbreakers = random.sample(DEALBREAKERS, k=random.randint(1,2))
    frustration = random.choice(["low","medium","high"])
    procrast = random.choice(["low","medium","high"])

    p = PersonalityProfile(
        positive_traits=pos, negative_traits=neg, coping_style=coping, conflict_style=conflict,
        decision_style=decision, trust_sales=trust_sales, trust_clinical=trust_clinical,
        skepticism_triggers=skept_trig, motivations=motivations, fears=fears, dealbreakers=dealbreakers,
        frustration_threshold=frustration, procrastination_tendency=procrast,
        **bigfive
    )
    # Ensure at least one clearly negative angle: high neuroticism or low trust_sales increases cancellation risk downstream
    return p

def _pick_life_context() -> LifeContext:
    employment = random.choices(["full_time","part_time","unemployed","retired","student"], weights=[0.45,0.2,0.15,0.1,0.1])[0]
    schedule = random.choice(["fixed_shifts","flexible","gig","unemployed"])
    caregiver = random.choices(["none","children","elderly","both"], weights=[0.5,0.25,0.15,0.1])[0]
    transport = random.choices(["car","public_transit","rideshare_only","none"], weights=[0.6,0.25,0.1,0.05])[0]
    distance = random.randint(5, 90)
    budget_sens = random.choices(["low","medium","high"], weights=[0.2,0.5,0.3])[0]
    budget_cap = random.choice([2500, 4000, 5000, 8000, 12000])
    prev_neg = random.sample(NEG_EXPERIENCES, k=random.randint(0,2))
    barriers = random.sample(BARRIERS, k=random.randint(1,3))
    return LifeContext(employment_status=employment, work_schedule_constraint=schedule, caregiver_responsibility=caregiver,
                       transport_access=transport, distance_to_center_minutes=distance, budget_sensitivity=budget_sens,
                       budget_cap_usd=budget_cap, previous_negative_experiences=prev_neg, barriers=barriers)

def _pick_compliance(personality: PersonalityProfile, life: LifeContext) -> ComplianceBehavior:
    base_adherence = 70
    base_cancel = 15
    no_show = "medium"
    price_sens = life.budget_sensitivity
    skept = "medium"

    # Heuristics: higher neuroticism + avoidant coping -> more cancellations
    if personality.neuroticism >= 4 or personality.coping_style == "avoidant":
        base_cancel += 10
        base_adherence -= 10
        no_show = "high"
    # Low trust in sales/medical increases skepticism
    if personality.trust_sales == "low" or personality.trust_clinical == "low":
        skept = "high"
        base_adherence -= 5
    # Long distance and transport barriers -> more cancellations
    if life.distance_to_center_minutes > 45 or "transport" in life.barriers:
        base_cancel += 10
        no_show = "high"

    # Clip probabilities
    base_adherence = max(0, min(100, base_adherence))
    base_cancel = max(0, min(100, base_cancel))

    followup = random.choice(["low","medium","high"])
    financing_comfort = random.choices(["low","medium","high"], weights=[0.35,0.45,0.2])[0]
    pain_tol = random.choice(["low","medium","high"])

    return ComplianceBehavior(
        adherence_probability=base_adherence,
        cancellation_probability=base_cancel,
        no_show_risk=no_show,
        price_sensitivity=price_sens,
        financing_comfort=financing_comfort,
        pain_tolerance=pain_tol,
        skepticism_level=skept,
        follow_up_responsiveness=followup,
    )

# ---------- Generator ----------
def generate_dental_persona() -> 'DentalPersona':
    # Sample a human-friendly intent
    intent: FlowIntent = random.choices(
        [
            "schedule_consultation",
            "check_eligibility",
            "ask_insurance_and_financing",
            "learn_treatment_options",
        ],
        weights=[0.4, 0.25, 0.2, 0.15],
    )[0]
    tone = random.choice(["hopeful","anxious","curious","impatient","polite","decisive"])
    language_proficiency = random.choices(["fluent","ESL_mild"], weights=[0.85,0.15])[0]
    prefers_human_agent = random.choices(["low","medium","high"], weights=[0.7,0.2,0.1])[0]
    filler_tolerance = random.choice(["low","medium","high"])

    ident = _pick_identity_flat()
    cc, phone = rand_phone()
    email = rand_email(ident["first_name"], ident["last_name"])

    # Heuristic: callers wanting to schedule are more likely in a hurry
    in_a_hurry = tone in ("impatient","decisive") or (intent == "schedule_consultation" and random.random() < 0.4)
    # Category 2 assumes consult-oriented calls, keep for_consult=True
    condition = _pick_condition(for_consult=True)
    appointment = _pick_appointment(in_a_hurry=in_a_hurry)
    address = _pick_address()
    financing = _pick_financing(condition)
    # Nudge fields to be intent-consistent
    if intent in ("ask_insurance_and_financing", "learn_treatment_options"):
        financing.interested_in_financing = True
        # Make score band known more often to exercise HCF branches
        financing.credit_score_band = random.choices(["650_or_higher","below_650","unknown"], weights=[0.45,0.4,0.15])[0]
        # If score <650, hint cosigner
        if financing.credit_score_band == "below_650" and not financing.has_cosigner_available:
            financing.has_cosigner_available = random.random() < 0.6
        # Insurance transparency is relevant regardless of has_insurance; leave as generated
    elif intent == "check_eligibility":
        # Keep natural distribution but ensure at least one hard-check signal is present
        if not condition.missing_broken_failing and not condition.gum_disease:
            condition.missing_broken_failing = True
        # Keep financing interest optional here
    elif intent == "schedule_consultation":
        # Lean toward accepting earliest slot and having appointment prefs solid
        appointment.will_accept_earliest_slot = appointment.will_accept_earliest_slot or (random.random() < 0.3)
        # Encourage accompaniment mention downstream
        accompaniment_planned = True if random.random() < 0.7 else False
    # Sample an optional handoff trigger (35% of personas)
    handoff = HandoffContext()
    if random.random() < 0.35:
        possible_triggers: List[HandoffTrigger] = [
            "verify_appointment_datetime","running_late","cannot_reach_center","no_confirmation_call",
            "existing_implant_repair","traveling_implant_damage","mend_troubleshooting",
            "online_or_virtual_request","needs_prescription","refund_request","fraudulent_loan",
            "third_party_confirm_appt","attorney_office_caller","hospital_or_dentist_office",
            "scan_copy_cost","legal_mail_address","outbound_number_spam","second_or_repeat_consult",
            "double_consult_request","did_not_schedule","accessibility_blind_deaf","asl_interpreter_virtual",
            "pregnancy","under_18_dob",
        ]
        trig = random.choice(possible_triggers)
        handoff.trigger = trig
        if trig in {"verify_appointment_datetime","running_late","no_confirmation_call","did_not_schedule"}:
            handoff.is_existing_patient = True
            handoff.has_existing_appointment = True
            handoff.verify_appointment = trig == "verify_appointment_datetime"
            handoff.running_late = trig == "running_late"
            if trig == "did_not_schedule":
                handoff.did_not_schedule = True
        if trig in {"existing_implant_repair","traveling_implant_damage"}:
            handoff.is_existing_patient = True
            handoff.existing_implant_repair = trig == "existing_implant_repair"
            handoff.traveling_implant_damage = trig == "traveling_implant_damage"
        if trig == "cannot_reach_center":
            handoff.is_existing_patient = True
            handoff.cannot_reach_center = True
        if trig == "mend_troubleshooting":
            handoff.is_existing_patient = True
        if trig == "online_or_virtual_request":
            handoff.wants_virtual_only = True
        if trig == "needs_prescription":
            handoff.needs_prescription = True
        if trig == "refund_request":
            handoff.refund_request = True
        if trig == "fraudulent_loan":
            handoff.finance_issue = "fraudulent_loan"
        if trig == "third_party_confirm_appt":
            handoff.third_party_role = "transport_or_insurance"
        if trig == "attorney_office_caller":
            handoff.third_party_role = "attorney"
        if trig == "hospital_or_dentist_office":
            handoff.third_party_role = "hospital_or_dentist"
        if trig == "scan_copy_cost":
            handoff.info_request = "scan_copy_cost"
        if trig == "legal_mail_address":
            handoff.info_request = "legal_mail_address"
        if trig == "outbound_number_spam":
            handoff.outbound_number_spam = True
        if trig == "second_or_repeat_consult":
            handoff.second_or_repeat_consult = True
        if trig == "double_consult_request":
            handoff.double_consult_request = True
        if trig == "accessibility_blind_deaf":
            handoff.accessibility_need = random.choice(["blind","deaf"])
        if trig == "asl_interpreter_virtual":
            handoff.asl_for_virtual = True
            handoff.accessibility_need = "asl"
        if trig == "pregnancy":
            handoff.pregnancy = True
        if trig == "under_18_dob":
            handoff.under_18 = True
            young_year = date.today().year - random.randint(12, 17)
            ident["date_of_birth"] = f"{young_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    personality = _pick_personality()
    life = _pick_life_context()
    compliance = _pick_compliance(personality, life)

    # If set above for inbound intent, keep; else sample default
    try:
        accompaniment_planned
    except NameError:
        accompaniment_planned = random.random() < 0.6
    accompaniment_name = f"{rand_name()[0]} {rand_name()[1]}" if accompaniment_planned else ""

    return DentalPersona(
        intent=intent,
        tone=tone, language_proficiency=language_proficiency,
        prefers_human_agent=prefers_human_agent, allow_human_anytime=True,
        in_a_hurry=in_a_hurry, filler_tolerance=filler_tolerance,
        title=ident["title"], first_name=ident["first_name"], last_name=ident["last_name"],
        gender=ident["gender"], date_of_birth=ident["date_of_birth"], dob_provided=ident["dob_provided"],
        phone_country_code=cc, phone_number_only=phone, email_address=email,
        preferred_notification_channel=random.choices(["email","sms","both"], weights=[0.35,0.25,0.4])[0],
        address=address, condition=condition, appointment=appointment, insurance_financing=financing,
        personality=personality, life_context=life, compliance=compliance,
        accompaniment_planned=accompaniment_planned, accompaniment_name=accompaniment_name,
        privacy_policy_ack=True,
        handoff=handoff,
    )

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _persona_filename(first_name: str, last_name: str, pid: int) -> str:
    return f"{first_name}_{last_name}_P{pid}.json"

if __name__ == "__main__":
    out_dir = os.path.join("user_persona", "dental")
    count = 100
    _ensure_dir(out_dir)
    for pid in range(1, count + 1):
        persona = generate_dental_persona()
        data = asdict(persona)
        data["id"] = pid
        fname = _persona_filename(data.get("first_name","User"), data.get("last_name","Unknown"), pid)
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
