"""
Downloads the CLINC150 intent classification dataset and converts it into
flat train/val/test CSVs under data/processed/.

Source: https://github.com/clinc/oos-eval (Larson et al., 2019)
150 in-scope intents across 10 domains (banking, travel, utility, work,
small_talk, meta, credit_cards, home, auto, kitchen_dining) + an
out-of-scope (oos) class for "route to human / fallback".

Usage:
    python src/data/load_data.py
"""
import json
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Map CLINC150's fine-grained intents to coarse routing domains.
# This is what turns "150-class intent classification" into an actual
# "which downstream agent handles this" routing decision.
DOMAIN_MAP = {
    "banking": "billing_agent",
    "credit_cards": "billing_agent",
    "kitchen_dining": "general_agent",
    "home": "general_agent",
    "auto_and_commute": "general_agent",
    "travel": "general_agent",
    "utility": "tech_support_agent",
    "work": "general_agent",
    "small_talk": "chitchat_agent",
    "meta": "tech_support_agent",
}

# CLINC150's raw file doesn't ship domain labels per-example directly in
# data_full.json (those live in a separate mapping in the repo). To keep
# this self-contained we derive a domain from the intent name using the
# repo's published intent->domain grouping below.
INTENT_TO_DOMAIN = {
    # banking
    "transfer": "banking", "transactions": "banking", "balance": "banking",
    "freeze_account": "banking", "pay_bill": "banking", "bill_balance": "banking",
    "bill_due": "banking", "interest_rate": "banking", "routing": "banking",
    "min_payment": "banking", "order_checks": "banking", "pin_change": "banking",
    "report_fraud": "banking", "account_blocked": "banking", "spending_history": "banking",
    # credit_cards
    "credit_score": "credit_cards", "report_lost_card": "credit_cards",
    "credit_limit": "credit_cards", "rewards_balance": "credit_cards",
    "new_card": "credit_cards", "application_status": "credit_cards",
    "card_declined": "credit_cards", "international_fees": "credit_cards",
    "apr": "credit_cards", "redeem_rewards": "credit_cards",
    "credit_limit_change": "credit_cards", "damaged_card": "credit_cards",
    "replacement_card_duration": "credit_cards", "improve_credit_score": "credit_cards",
    "expiration_date": "credit_cards",
    # travel
    "book_flight": "travel", "book_hotel": "travel", "car_rental": "travel",
    "travel_suggestion": "travel", "travel_alert": "travel", "travel_notification": "travel",
    "carry_on": "travel", "timezone": "travel", "vaccines": "travel",
    "translate": "travel", "flight_status": "travel", "international_visa": "travel",
    "lost_luggage": "travel", "plug_type": "travel", "exchange_rate": "travel",
    # utility
    "alarm": "utility", "timer": "utility", "share_location": "utility",
    "text": "utility", "spelling": "utility", "calculator": "utility",
    "measurement_conversion": "utility", "flip_coin": "utility", "roll_dice": "utility",
    "definition": "utility", "make_call": "utility", "date": "utility",
    "weather": "utility", "todo_list_update": "utility", "reminder": "utility",
    # work
    "pto_request": "work", "pto_used": "work", "next_holiday": "work",
    "pto_balance": "work", "pto_request_status": "work", "meeting_schedule": "work",
    "schedule_meeting": "work", "payday": "work", "taxes": "work",
    "income": "work", "rollover_401k": "work", "w2": "work",
    "schedule_maintenance": "work", "direct_deposit": "work", "insurance_change": "work",
    # small_talk
    "greeting": "small_talk", "goodbye": "small_talk", "tell_joke": "small_talk",
    "where_are_you_from": "small_talk", "how_old_are_you": "small_talk",
    "what_is_your_name": "small_talk", "who_made_you": "small_talk",
    "thank_you": "small_talk", "what_can_i_ask_you": "small_talk",
    "what_are_your_hobbies": "small_talk", "do_you_have_pets": "small_talk",
    "are_you_a_bot": "small_talk", "meaning_of_life": "small_talk",
    "who_do_you_work_for": "small_talk", "fun_fact": "small_talk",
    # meta
    "change_ai_name": "meta", "change_user_name": "meta", "cancel": "meta",
    "user_name": "meta", "reset_settings": "meta", "whisper_mode": "meta",
    "repeat": "meta", "no": "meta", "yes": "meta", "maybe": "meta",
    "change_language": "meta", "change_accent": "meta", "change_volume": "meta",
    "change_speed": "meta", "sync_device": "meta",
    # home
    "shopping_list": "home", "shopping_list_update": "home", "next_song": "home",
    "play_music": "home", "update_playlist": "home", "order": "home",
    "order_status": "home", "food_last": "home", "ingredients_list": "home",
    "recipe": "home", "cook_time": "home", "calories": "home",
    "nutrition_info": "home", "restaurant_reservation": "home", "restaurant_reviews": "home",
    "restaurant_suggestion": "home", "smart_home": "home", "gas": "home",
    "gas_type": "home", "distance": "home",
    # auto_and_commute
    "traffic": "auto_and_commute", "directions": "auto_and_commute",
    "gas_type": "auto_and_commute", "oil_change_when": "auto_and_commute",
    "oil_change_how": "auto_and_commute", "current_location": "auto_and_commute",
    "tire_pressure": "auto_and_commute", "tire_change": "auto_and_commute",
    "jump_start": "auto_and_commute", "mpg": "auto_and_commute",
    "uber": "auto_and_commute", "schedule_maintenance": "auto_and_commute",
    "last_maintenance": "auto_and_commute", "vehicle_service": "auto_and_commute",
    # kitchen_dining -> reuse recipe/cook_time already mapped under home;
    # left intentionally minimal since CLINC intents overlap categories.
}


def download_raw(force: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "clinc150_data_full.json"

    if raw_path.exists() and not force:
        logger.info(f"Raw file already present at {raw_path}, skipping download.")
    else:
        logger.info(f"Downloading CLINC150 from {DATA_URL} ...")
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()
        raw_path.write_bytes(resp.content)
        logger.info(f"Saved raw data to {raw_path} ({len(resp.content) / 1024:.1f} KB)")

    return json.loads(raw_path.read_text())


def to_dataframe(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=["text", "intent"])
    df["domain"] = df["intent"].map(INTENT_TO_DOMAIN).fillna("other")
    df["route"] = df["intent"].apply(
        lambda i: "fallback_human" if i == "oos" else DOMAIN_MAP.get(
            INTENT_TO_DOMAIN.get(i, "other"), "general_agent"
        )
    )
    return df


def main():
    raw = download_raw()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, out_name in [
        ("train", "train.csv"),
        ("val", "val.csv"),
        ("test", "test.csv"),
    ]:
        in_scope = raw[split_name]
        oos_key = f"oos_{split_name}"
        oos = raw.get(oos_key, [])
        combined = in_scope + oos

        df = to_dataframe(combined)
        out_path = PROCESSED_DIR / out_name
        df.to_csv(out_path, index=False)
        logger.info(
            f"{split_name}: {len(df)} rows -> {out_path} "
            f"({df['intent'].nunique()} intents, {df['route'].nunique()} routes)"
        )

    logger.info("Route distribution (train):")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    logger.info("\n" + train_df["route"].value_counts().to_string())


if __name__ == "__main__":
    main()
