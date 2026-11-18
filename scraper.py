import os
import re
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Literal, Set
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Απενεργοποίηση προειδοποιήσεων για sites με χαλασμένο SSL (π.χ. Chronolog)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TARGET_SOURCES = [
    "https://www.runningnews.gr/category.php?cat=9",              # Δρόμος
    "https://www.runningnews.gr/category.php?subcat=44",          # Βουνό
    "https://www.advendure.com/index.php/calendar",               # Advendure 
    "https://runnermagazine.gr/category/kalentari/",              # Runner Magazine (Διορθωμένο)
    "https://myathlete.gr/events/",                               # MyAthlete (Χωρίς www)
    "https://chronolog.gr/events/",                               # Chronolog
    "https://aims-worldrunning.org/calendar.html",                # AIMS
]

STOP_WORDS = {
    "αγωνας", "αγωνες", "δρομου", "βουνου", "ορεινος", "ορεινου",
    "race", "run", "running", "trail", "mountain", "ultra", "the",
    "του", "της", "των", "τον", "την", "στο", "στη", "στον", "στην", "και", "με"
}

class RaceItem(BaseModel):
    title: str = Field(description="Καθαρός τίτλος του αγώνα")
    date: str = Field(description="Ημερομηνία διεξαγωγής αυστηρά σε μορφή DD/MM/YYYY")
    location: str = Field(description="Πόλη, βουνό ή χώρα διεξαγωγής")
    terrain: Literal["Δρόμος", "Βουνό", "Μικτό", "Στίβος"] = Field(description="Κατηγορία τερέν")
    distances: str = Field(description="Διαθέσιμες αποστάσεις χωρισμένες με κόμμα")
    link: str = Field(description="Πλήρες URL της σελίδας του αγώνα")

class ExtractedRaces(BaseModel):
    races: List[RaceItem]

def normalize_text(text: str) -> str:
    nfd = unicodedata.normalize("NFD", str(text or "").lower())
    no_accents = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    no_ordinals = re.sub(r"\b\d{1,3}(ος|η|ο|ου|ης|ων|th|st|nd|rd)\b", " ", no_accents)
    no_years = re.sub(r"\b20\d{2}\b", " ", no_ordinals)
    clean_chars = re.sub(r"[^a-zα-ω0-9\s]", " ", no_years)
    tokens = [t for t in clean_chars.split() if len(t) >= 3 and t not in STOP_WORDS]
    return " ".join(tokens) if tokens else " ".join(clean_chars.split())

def is_duplicate_race(candidate: dict, seen_by_date: Dict[str, List[dict]]) -> bool:
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
        if cand_norm and cand_norm == ex_norm: return True
        if len(cand_norm) >= 6 and len(ex_norm) >= 6 and (cand_norm in ex_norm or ex_norm in cand_norm): return True
        if cand_tokens and ex_tokens:
            common = len(cand_tokens & ex_tokens)
            min_len = min(len(cand_tokens), len(ex_tokens))
            if common / min_len >= 0.65: return True
            if common >= 1 and len(cand_loc) >= 4 and len(ex_loc) >= 4 and (cand_loc in ex_loc or ex_loc in cand_loc): return True
    return False

def fetch_clean_page_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    response = None
    for attempt in range(2):
        try:
            # verify=False παρακάμπτει το ληγμένο SSL της Chronolog
            response = requests.get(url, headers=headers, timeout=25, verify=False)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 1: raise
            time.sleep(2)

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "svg", "noscript"]):
        tag.decompose()

    ignored_domains = ("facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com")
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")): continue
        full_url = urljoin(url, href)
        if any(dom in full_url.lower() for dom in ignored_domains):
            a_tag.decompose()
            continue
        link_text = a_tag.get_text(" ", strip=True)
        if link_text and full_url.startswith(("http://", "https://")):
            a_tag.replace_with(f"{link_text} (URL: {full_url})")

    seen_lines = set()
    clean_lines = []
    for raw_line in soup.get_text("\n").splitlines():
        line = " ".join(raw_line.split())
        if len(line) >= 3 and line not in seen_lines:
            seen_lines.add(line)
            clean_lines.append(line)
    return "\n".join(clean_lines)[:60000]

def chunk_text(text: str, max_chars: int = 30000) -> List[str]:
    # Αυξήσαμε τα chars σε 30k για να γλιτώνουμε κλήσεις (API Quota)
    chunks, current_lines, current_len = [], [], 0
    for line in text.splitlines():
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines, current_len = [line], line_len
        else:
            current_lines.append(line)
            current_len += line_len
    if current_lines: chunks.append("\n".join(current_lines))
    return chunks

def extract_races_from_chunk(client: genai.Client, source_url: str, text_chunk: str) -> List[dict]:
    today_str = datetime.now().strftime("%d/%m/%Y")
    prompt = f"""Είσαι βοηθός εξαγωγής δεδομένων. Σημερινή ημερομηνία: {today_str}. Πηγή: {source_url}
Ανάλυσε το απόσπασμα και εξήγαγε τους επερχόμενους αγώνες (από {today_str} και μετά).
1. date: αυστηρά DD/MM/YYYY. Αν λείπει το έτος, υπολόγισε το σωστό.
2. terrain: "Δρόμος", "Βουνό", "Μικτό", ή "Στίβος".
3. link: το URL του αγώνα αν υπάρχει, αλλιώς το {source_url}.

Κείμενο:
{text_chunk}"""

    # Βάλαμε 3.8-flash ως πρώτο γιατί είναι ταχύτερο και αντέχει μεγαλύτερα chunks χωρίς 503
    models_to_try = ["gemini-3.8-flash", "gemini-4-hydrogen"]
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
            if any(code in error_text for code in ("503", "UNAVAILABLE", "429")) and attempt < len(models_to_try) - 1:
                print(f"   Φόρτος στο {model_name}. Επανάληψη σε 10s με εναλλακτικό μοντέλο...")
                time.sleep(10)
            else:
                raise
    if not response or not response.parsed: return []
    return [race.model_dump() for race in response.parsed.races]

def send_to_google_sheets(races: List[dict], api_url: str, api_secret: str) -> None:
    response = requests.post(
        api_url,
        json={"secret": api_secret, "races": races},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    print("Απάντηση Google Sheets API:", response.text)

def main() -> None:
    google_api_url = os.environ.get("GOOGLE_API_URL", "").strip()
    ingest_api_secret = os.environ.get("INGEST_API_SECRET", "").strip()
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not google_api_url or not ingest_api_secret or not gemini_api_key: return

    client = genai.Client(api_key=gemini_api_key)
    seen_by_date: Dict[str, List[dict]] = {}
    total_sent = 0

    for url in TARGET_SOURCES:
        try:
            print(f"\n--- Σάρωση πηγής: {url} ---")
            clean_text = fetch_clean_page_text(url)
            chunks = chunk_text(clean_text, max_chars=30000)
            print(f"Διαχωρισμός σε {len(chunks)} τμήματα...")
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
                                "norm_title": norm_t, "tokens": set(norm_t.split()), "norm_loc": normalize_text(race.get("location", ""))
                            })
                            source_unique_races.append(race)
                    # Μικρή καθυστέρηση για αποφυγή rate limiting
                    time.sleep(4)
                except Exception as chunk_exc:
                    print(f"  Σφάλμα στο τμήμα {idx}/{len(chunks)}: {chunk_exc}")

            print(f"Βρέθηκαν {len(source_unique_races)} μοναδικοί αγώνες από {url}")
            if source_unique_races:
                send_to_google_sheets(source_unique_races, google_api_url, ingest_api_secret)
                total_sent += len(source_unique_races)

        except Exception as exc:
            print(f"Σφάλμα στη σελίδα {url}: {exc}")

    print(f"\nΟλοκλήρωση. Συνολικοί αγώνες: {total_sent}")

if __name__ == "__main__":
    main()
