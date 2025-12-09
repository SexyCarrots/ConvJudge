import argparse
import random
import string
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import List, Literal
import os
import json

# ---------- Enumerations ----------
CabinClass = Literal["Economy", "Premium Economy", "Aether Business"]
TripType = Literal["one_way", "round_trip", "multi_city"]
NotifChannel = Literal["email", "sms", "both"]
SeatType = Literal["standard", "extra_legroom", "exit_row", ""]
FlowIntent = Literal["new_booking", "change_booking", "info_only"]
LoyaltyTier = Literal["None", "Silver", "Gold", "Platinum", "Zenith"]
# Conversation behavior enums
ChatTone = Literal["polite", "impatient", "anxious", "confident", "chatty", "curt"]
LanguageProficiency = Literal["fluent", "ESL_mild"]
AgentPreference = Literal["low", "medium", "high"]

# ---------- Helpers ----------
AIRPORTS = ["SFO", "JFK", "LAX", "ORD", "DFW", "SEA", "BOS", "MIA", "IAD", "CDG", "LHR", "NRT", "DXB", "MAD"]
DOMESTIC = {"SFO","LAX","SEA","BOS","MIA","DFW","ORD","JFK","IAD"}
INTERNATIONAL = {"CDG","LHR","NRT","DXB","MAD"}
TITLES = ["Mr", "Ms", "Mx"]
GENDERS = ["M", "F", "X"]
NATIONALITIES = ["US","CA","GB","FR","ES","DE","JP","AE"]

def rand_name():
    first_pool = ["Alex","Taylor","Jordan","Casey","Morgan","Avery","Riley","Sofia","Liam","Noah","Mia","Ethan","Aiden","Zoe"]
    last_pool  = ["Smith","Johnson","Brown","Garcia","Miller","Davis","Martinez","Lopez","Wilson"]
    return random.choice(first_pool), random.choice(last_pool)

def rand_phone():
    return "+1", "".join(random.choices(string.digits, k=10))

def rand_email(first, last):
    domains = ["example.com","mail.com","inbox.dev","test.org"]
    return f"{first.lower()}.{last.lower()}@{random.choice(domains)}"

def future_date(days_min=7, days_max=120):
    return date.today() + timedelta(days=random.randint(days_min, days_max))

def passport_number():
    return random.choice(string.ascii_uppercase) + "".join(random.choices(string.digits, k=7))

def last16():
    return "".join(random.choices(string.digits, k=16))

def random_time_window():
    buckets = ["early morning","morning","midday","afternoon","evening","13:00-17:00","18:00-22:00"]
    return random.choice(buckets)

def pick_legs(trip_type: TripType):
    origin = random.choice(list(DOMESTIC))
    dest = random.choice([a for a in AIRPORTS if a != origin])
    out_date = future_date()
    legs = [{"origin": origin, "destination": dest, "departure_date": str(out_date), "preferred_time_window": random_time_window()}]
    if trip_type in ("round_trip","multi_city"):
        back_origin = dest
        back_dest = origin
        back_date = out_date + timedelta(days=random.randint(2, 14))
        legs.append({"origin": back_origin, "destination": back_dest, "departure_date": str(back_date), "preferred_time_window": random_time_window()})
    if trip_type == "multi_city":
        hub = random.choice([a for a in AIRPORTS if a not in {origin, dest}])
        legs.append({"origin": back_dest, "destination": hub, "departure_date": str(back_date + timedelta(days=random.randint(2,7))), "preferred_time_window": random_time_window()})
    return legs

def is_international(legs):
    def zone(a):
        if a in DOMESTIC: return "dom"
        if a in INTERNATIONAL: return "int"
        return "dom"
    for leg in legs:
        if zone(leg["origin"]) != zone(leg["destination"]):
            return True
    return False

# ---------- Data classes (no Optional types) ----------
@dataclass
class Passenger:
    title: str
    first_name: str
    last_name: str
    gender: str
    date_of_birth: str  # YYYY-MM-DD
    nationality: str
    passport_number: str = ""           # "" when not applicable
    passport_expiration_date: str = ""  # "" when not applicable
    known_traveler_number: str = ""     # "" when not provided
    redress_number: str = ""            # "" when not provided

@dataclass
class Ancillaries:
    seat_selection: SeatType = ""    # "" when none selected
    checked_bags: int = 0
    sports_equipment: str = ""       # "" when none
    special_meal: str = ""           # "" when none
    lounge_access: bool = False
    priority_boarding: bool = False
    wifi: bool = False
    carbon_offset: bool = False

@dataclass
class PaymentPrefs:
    payment_method_type: str = ""    # "card"|"wallet"|"voucher"|"" (info-only)
    card_num: str = ""             # "" if voucher/info-only
    consent_to_secure_capture: bool = True

@dataclass
class AirlinePersona:
    # Flow selection
    intent: FlowIntent
    # Conversation behavior
    tone: ChatTone
    language_proficiency: LanguageProficiency
    prefers_human_agent: AgentPreference
    # Lookup & account
    phone_country_code: str
    phone_number_only: str
    loyalty_number: str = ""         # "" if none
    loyalty_tier: LoyaltyTier = "None"
    # Greeting fallback
    first_name: str = ""
    last_name: str = ""
    # Verification
    otp_6digit: str = ""             # "" if not provided
    # Trip planning
    trip_type: TripType = "one_way"
    cabin_class: CabinClass = "Economy"
    legs: List[dict] = field(default_factory=list)
    # Passengers
    passengers: List[Passenger] = field(default_factory=list)
    # Contact & delivery
    email_address: str = ""
    preferred_notification_channel: NotifChannel = "both"
    # Ancillaries
    ancillaries: Ancillaries = field(default_factory=Ancillaries)
    # Change-booking specific
    existing_pnr: str = ""           # "" if not change_booking
    # Payment
    payment: PaymentPrefs = field(default_factory=PaymentPrefs)

# ---------- Generator ----------
def generate_airline_persona() -> AirlinePersona:
    # Intent
    intent = random.choices(["new_booking","change_booking","info_only"], weights=[0.6, 0.3, 0.1])[0]
    # Conversation behavior attributes
    tone = random.choice(["polite", "impatient", "anxious", "confident", "chatty", "curt"])
    language_proficiency = random.choices(["fluent", "ESL_mild"], weights=[0.8, 0.2])[0]
    prefers_human_agent = random.choices(["low", "medium", "high"], weights=[0.6, 0.3, 0.1])[0]
    # Lookup fields
    cc, phone = rand_phone()
    loyalty_tier = random.choices(["None","Silver","Gold","Platinum","Zenith"], weights=[0.5,0.2,0.15,0.1,0.05])[0]
    loyalty_number = "" if loyalty_tier == "None" else f"STRAT-{random.randint(10000,99999)}"
    # Name + email
    fn, ln = rand_name()
    email = rand_email(fn, ln)

    # OTP behavior
    otp_code = "".join(random.choices(string.digits, k=6))

    # Trip planning
    trip_type = random.choices(["one_way","round_trip","multi_city"], weights=[0.4,0.5,0.1])[0]
    cabin = random.choices(["Economy","Premium Economy","Aether Business"], weights=[0.7,0.2,0.1])[0]
    legs = pick_legs(trip_type)

    # Passengers
    pax_count = random.choices([1,2,3],[0.7,0.2,0.1])[0]
    pax = []
    # need_passport = is_international(legs)
    for i in range(pax_count):
        pfn, pln = (fn, ln) if i==0 else rand_name()
        dob_year = random.randint(1958, 2007)
        dob = f"{dob_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        nationality = random.choice(NATIONALITIES)
        p = Passenger(
            title=random.choice(TITLES),
            first_name=pfn,
            last_name=pln,
            gender=random.choice(GENDERS),
            date_of_birth=dob,
            nationality=nationality
        )
        exp_year = random.randint(date.today().year+1, date.today().year+10)
        p.passport_number = passport_number()
        p.passport_expiration_date = f"{exp_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        if random.random() < 0.3:
            p.known_traveler_number = "".join(random.choices(string.digits, k=9))
        if random.random() < 0.1:
            p.redress_number = "".join(random.choices(string.digits, k=7))
        pax.append(p)

    # Contact channel
    notif = random.choices(["email","sms","both"], weights=[0.25,0.25,0.5])[0]

    # Ancillaries via bundle
    bundle = random.choices(["none","light","comfort","businesslike"], [0.25,0.35,0.3,0.1])[0]
    anc = Ancillaries()
    if bundle in ("light","comfort","businesslike"):
        anc.checked_bags = random.choice([1,2])
    if bundle in ("comfort","businesslike"):
        anc.seat_selection = random.choice(["extra_legroom","exit_row","standard"])
        anc.priority_boarding = True
        anc.wifi = random.random() < 0.7
    if bundle == "businesslike":
        anc.lounge_access = True
        anc.special_meal = random.choice(["vegetarian","gluten_free","kosher","halal","diabetic"])
        anc.carbon_offset = random.random() < 0.4
    if random.random() < 0.1:
        anc.sports_equipment = random.choice(["golf bag","ski bag","bicycle"])

    # Payment prefs (meta only; full PAN via keypad at runtime)
    pay = PaymentPrefs()
    if intent in ("new_booking","change_booking"):
        pay.payment_method_type = random.choice(["card","wallet","voucher"])
        pay.card_num = last16() if pay.payment_method_type in ("card","wallet") else ""
        pay.consent_to_secure_capture = True
    else:
        # info_only: ensure safe defaults
        pay.payment_method_type = ""
        pay.card_num = ""
        pay.consent_to_secure_capture = True

    # Change booking specifics
    pnr = ""
    if intent == "change_booking":
        pnr = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    persona = AirlinePersona(
        first_name=fn,
        last_name=ln,
        intent=intent,
        tone=tone,
        language_proficiency=language_proficiency,
        prefers_human_agent=prefers_human_agent,
        phone_country_code=cc,
        phone_number_only=phone,
        loyalty_number=loyalty_number,
        loyalty_tier=loyalty_tier,
        otp_6digit=otp_code,
        trip_type=trip_type,
        cabin_class=cabin,
        legs=legs,
        passengers=pax,
        email_address=email,
        preferred_notification_channel=notif,
        ancillaries=anc,
        existing_pnr=pnr,
        payment=pay
    )
    return persona

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _persona_filename(first_name: str, last_name: str, pid: int) -> str:
    return f"{first_name}_{last_name}_{pid}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Celestar Air caller personas.")
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of personas to generate (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("user_persona", "airline_500"),
        help="Directory to store persona JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


# Batch generation main
if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    _ensure_dir(args.output_dir)

    for pid in range(1, args.count + 1):
        persona = generate_airline_persona()
        data = asdict(persona)
        data["id"] = pid
        fname = _persona_filename(data.get("first_name", "User"), data.get("last_name", "Unknown"), "P" + str(pid))
        fpath = os.path.join(args.output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
