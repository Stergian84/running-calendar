import os
import re
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Literal, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Διευρυμένη λίστα πηγών (Ελλάδα & Διεθνείς αγώνες)
TARGET_SOURCES = [
    "https://www.runningnews.gr/category.php?cat=9",              # RunningNews - Δρόμος
    "https://www.runningnews.gr/category.php?subcat=44",          # RunningNews - Βουνό
    "https://www.advendure.com/index.php/calendar",     # Advendure - Trail & Περιπέτεια
    "https://www.runnermagazine.gr/races/",                       # Runner Magazine - Καλεντάρι
    "https://www.myathlete.gr/events/",                           # MyAthlete - Επίσημοι Αγώνες
    "https://chronolog.gr/events/",                               # Chronolog - Αγώνες
    "https://aims-worldrunning.org/calendar.html",                # AIMS - Διεθνείς Μαραθώνιοι
]

STOP_WORDS = {
    "αγωνας", "αγωνες", "δρομου", "βουνου", "ορεινος", "ορεινου",
    "race", "run", "running", "trail", "mountain", "ultra", "the",
    "του", "της", "των", "τον", "την", "στο", "στη", "στον", "στην", "και", "με"
}


class RaceItem(BaseModel):
    title: str = Field(description="Καθαρός τίτλος του αγώνα χωρίς περιττά σύμβολα")
    date: str = Field(description="Ημερομηνία διεξαγωγής αυστηρά σε μορφή DD/MM/YYYY")
    location: str = Field(description="Πόλη, βουνό ή χώρα διεξαγωγής")
    terrain: Literal["Δρόμος", "Βουνό", "Μικτό", "Στίβος"] = Field(
        description="Κατηγορία τερέν του αγώνα"
    )
    distances: str = Field(
        description="Διαθέσιμες αποστάσεις χωρισμένες με κόμμα, π.χ. 21km, 10km, 5km"
    )
    link: str = Field(
        description="Πλήρες URL (https://...) της σελίδας πληροφοριών ή εγγραφής του αγώνα"
    )


class ExtractedRaces(BaseModel):
    races: List[RaceItem]


def normalize_text(text: str) -> str:
    """Αφαιρεί τόνους, αύξοντες αριθμούς, έτη και γενικές λέξεις για έξυπνο έλεγχο διπλοεγγραφών."""
    nfd = unicodedata.normalize("NFD", str(text or "").lower())
    no_accents = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    no_ordinals = re.sub(r"\b\d{1,3}(ος|η|ο|ου|ης|ων|th|st|nd|rd)\b", " ", no_accents)
    no_years = re.sub(r"\b20\d{2}\b", " ", no_ordinals)
    clean_chars = re.sub(r"[^a-zα-ω0-9\s]", " ", no_years)
    tokens = [t for t in clean_chars.split() if len(t) >= 3 and t not in STOP_WORDS]
    return " ".join(tokens) if tokens else " ".join(clean_chars.split())


def is_duplicate_race(candidate: dict, seen_by_date: Dict[str, List[dict]]) -> bool:
    """Ελέγχει αν ο αγώνας υπάρχει ήδη την ίδια ημερομηνία από άλλη πηγή ή κομμάτι σελίδας."""
    date_str = candidate.get("date", "").strip()
    if not date_str or date_str not in seen_by_date:
        return False

    cand_norm = normalize_text(candidate.get("title", ""))
    cand_tokens: Set[str] = set(cand_norm.split())
    cand_loc = normalize_text(candidate.get("location", ""))

    for existing in seen_by_date[date_str]:
        ex_norm = existing["norm_title"]
        ex_tokens = existing["tokens"]
        ex_loc = existing["norm_loc"]

        if cand_norm and cand_norm == ex_norm:
            return True

        if len(cand_norm) >= 6 and len(ex_norm) >= 6 and (cand_norm in ex_norm or ex_norm in cand_norm):
            return True

        if cand_tokens and ex_tokens:
            common = len(cand_tokens & ex_tokens)
            min_len = min(len(cand_tokens), len(ex_tokens))
            if common / min_len >= 0.65:
                return True

            if common >= 1 and len(cand_loc) >= 4 and len(ex_loc) >= 4 and (cand_loc in ex_loc or ex_loc in cand_loc):
                return True

    return False


def fetch_clean_page_text(url: str) -> str:
    """Κατεβάζει την ιστοσελίδα, αφαιρεί τον θόρυβο και επιστρέφει καθαρό κείμενο με links."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }

    response = None
    for attempt in range(2):
        try:
            response = requests.get(url, headers=headers, timeout=25)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 1:
                raise
            time.sleep(2)

    soup = BeautifulSoup(response.text, "html.parser")

    # Αφαιρούμε μενού, banners και περιττά στοιχεία σελίδας
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "svg", "noscript"]):
        tag.decompose()

    ignored_domains = ("facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com", "linkedin.com")

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        full_url = urljoin(url, href)
        if any(dom in full_url.lower() for dom in ignored_domains):
            a_tag.decompose()
            continue
        link_text = a_tag.get_text(" ", strip=True)
        if link_text and full_url.startswith(("http://", "https://")):
            a_tag.replace_with(f"{link_text} (URL: {full_url})")

    # Αφαίρεση επαναλαμβανόμενων κενών ή πανομοιότυπων γραμμών
    seen_lines = set()
    clean_lines = []
    for raw_line in soup.get_text("\n").splitlines():
        line = " ".join(raw_line.split())
        if len(line) < 3:
            continue
        if line not in seen_lines:
            seen_lines.add(line)
            clean_lines.append(line)

    # Διαβάζουμε έως 45.000 χαρακτήρες ανά σελίδα (θα σπάσουν σε κομμάτια των 10.000)
    return "\n".join(clean_lines)[:45000]


def chunk_text(text: str, max_chars: int = 10000) -> List[str]:
    """Χωρίζει το κείμενο σε μικρά κομμάτια χωρίς να κόβει γραμμές στη μέση."""
    chunks = []
    current_lines = []
    current_len = 0

    for line in text.splitlines():
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def extract_races_from_chunk(client: genai.Client, source_url: str, text_chunk: str) -> List[dict]:
    """Εξάγει τους αγώνες από ένα κομμάτι κειμένου με γρήγορο fallback μοντέλων."""
    today_str = datetime.now().strftime("%d/%m/%Y")

    prompt = f"""
Είσαι βοηθός εξαγωγής δεδομένων για αγωνιστικό ημερολόγιο δρομέων.
Σημερινή ημερομηνία: {today_str}.
Πηγή σελίδας: {source_url}

Ανάλυσε το παρακάτω απόσπασμα ιστοσελίδας και εξήγαγε ΜΟΝΟ τους επερχόμενους αγώνες τρεξίματος (από {today_str} και μετά).

Κανόνες:
1. Η ημερομηνία (date) πρέπει να είναι αυστηρά σε μορφή DD/MM/YYYY (π.χ. 29/11/2026). Αν λείπει το έτος, υπολόγισε το σωστό με βάση τη σημερινή ημερομηνία ({today_str}).
2. Αν ένας αγώνας διαρκεί 2+ μέρες, βάλε την ημερομηνία έναρξης.
3. Στο terrain επίλεξε αυστηρά μία από τις τιμές: "Δρόμος", "Βουνό", "Μικτό", "Στίβος".
4. Στο link βάλε το συγκεκριμένο URL του αγώνα αν υπάρχει δίπλα του, αλλιώς το {source_url}.
5. Μην επινοείς αγώνες που δεν αναφέρονται ρητά.

Απόσπασμα σελίδας:
{text_chunk}
"""

    models_to_try = ["gemini-4-hydrogen", "gemini-4-hydrogen", "gemini-3.8-flash"]
    response = None

    for attempt, model_name in enumerate(models_to_try):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedRaces,
                    temperature=0.1,
                ),
            )
            break
        except Exception as exc:
            error_text = str(exc)
            is_temporary = any(code in error_text for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if is_temporary and attempt < len(models_to_try) - 1:
                wait_seconds = 8 * (attempt + 1)
                next_model = models_to_try[attempt + 1]
                print(f"   Φόρτος στο {model_name}. Επανάληψη σε {wait_seconds}s με {next_model}...")
                time.sleep(wait_seconds)
            else:
                raise

    if not response or not response.parsed:
        return []

    return [race.model_dump() for race in response.parsed.races]


def send_to_google_sheets(races: List[dict], api_url: str, api_secret: str) -> None:
    """Στέλνει τους μοναδικούς αγώνες στο Google Apps Script Web App."""
    # ΑΣΦΑΛΕΙΑ: Ο μυστικός κωδικός διαβάζεται μόνο από μεταβλητές περιβάλλοντος (GitHub Secrets).
    payload = {
        "secret": api_secret,
        "races": races,
    }
    response = requests.post(
        api_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    print("Απάντηση Google Sheets API:", response.text)


def main() -> None:
    google_api_url = os.environ.get("GOOGLE_API_URL", "").strip()
    ingest_api_secret = os.environ.get("INGEST_API_SECRET", "").strip()
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not google_api_url or not ingest_api_secret or not gemini_api_key:
        raise RuntimeError("Λείπουν απαραίτητες μεταβλητές περιβάλλοντος (Secrets).")

    client = genai.Client(api_key=gemini_api_key)
    seen_by_date: Dict[str, List[dict]] = {}
    total_sent = 0

    for url in TARGET_SOURCES:
        try:
            print(f"\n--- Σάρωση πηγής: {url} ---")
            clean_text = fetch_clean_page_text(url)
            chunks = chunk_text(clean_text, max_chars=10000)
            print(f"Διαχωρισμός σε {len(chunks)} τμήματα για σταθερή ανάλυση...")

            source_unique_races = []

            for idx, chunk in enumerate(chunks, start=1):
                try:
                    print(f"  Ανάλυση τμήματος {idx}/{len(chunks)}...")
                    chunk_races = extract_races_from_chunk(client, url, chunk)

                    for race in chunk_races:
                        if not is_duplicate_race(race, seen_by_date):
                            date_key = race.get("date", "").strip()
                            norm_t = normalize_text(race.get("title", ""))
                            seen_by_date.setdefault(date_key, []).append({
                                "norm_title": norm_t,
                                "tokens": set(norm_t.split()),
                                "norm_loc": normalize_text(race.get("location", "")),
                            })
                            source_unique_races.append(race)

                    time.sleep(2)
                except Exception as chunk_exc:
                    print(f"  Σφάλμα στο τμήμα {idx}/{len(chunks)}: {chunk_exc}")

            print(f"Βρέθηκαν {len(source_unique_races)} μοναδικοί αγώνες από {url}")

            if source_unique_races:
                send_to_google_sheets(source_unique_races, google_api_url, ingest_api_secret)
                total_sent += len(source_unique_races)

        except Exception as exc:
            print(f"Σφάλμα στη σελίδα {url}: {exc}")

    print(f"\nΟλοκλήρωση μαζικής αναζήτησης. Συνολικοί αγώνες που στάλθηκαν: {total_sent}")


if __name__ == "__main__":
    main()
