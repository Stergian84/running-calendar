import os
import time
from datetime import datetime
from typing import List, Literal
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Λίστα πηγών (URLs) προς σάρωση — μπορείς να προσθέσεις ή να αφαιρέσεις όποιο link θέλεις
TARGET_SOURCES = [
    "https://www.runningnews.gr/category.php?cat=9",              # RunningNews - Δρόμος
    "https://www.runningnews.gr/category.php?subcat=44",          # RunningNews - Βουνό
    "https://www.advendure.com/index.php/agonistikos-odigos",     # Advendure - Trail & Περιπέτεια
]


class RaceItem(BaseModel):
    title: str = Field(description="Καθαρός τίτλος του αγώνα χωρίς περιττά σύμβολα")
    date: str = Field(description="Ημερομηνία διεξαγωγής αυστηρά σε μορφή DD/MM/YYYY")
    location: str = Field(description="Πόλη, βουνό ή περιοχή διεξαγωγής στην Ελλάδα")
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


def fetch_clean_page_text(url: str) -> str:
    """Κατεβάζει την ιστοσελίδα και επιστρέφει καθαρό κείμενο διατηρώντας τους συνδέσμους."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
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

    # Αφαιρούμε στοιχεία που δεν περιέχουν αγώνες για εξοικονόμηση tokens
    for tag in soup(["script", "style", "nav", "footer", "svg", "noscript"]):
        tag.decompose()

    # Ενσωματώνουμε τα URLs δίπλα στα links ώστε να μπορεί να τα διαβάσει το Gemini
    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(url, a_tag["href"].strip())
        link_text = a_tag.get_text(" ", strip=True)
        if link_text and full_url.startswith(("http://", "https://")):
            a_tag.replace_with(f"{link_text} (URL: {full_url})")

    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    # Περιορίζουμε το μέγεθος στους πρώτους 35.000 χαρακτήρες ανά σελίδα
    return clean_text[:35000]


def extract_races_with_gemini(client: genai.Client, source_url: str, page_text: str) -> List[dict]:
    """Χρησιμοποιεί το Gemini API για δομημένη εξαγωγή αγώνων από το κείμενο της σελίδας."""
    today_str = datetime.now().strftime("%d/%m/%Y")

    prompt = f"""
Είσαι βοηθός εξαγωγής δεδομένων για αγωνιστικό ημερολόγιο δρομέων στην Ελλάδα.
Σημερινή ημερομηνία: {today_str}.
Πηγή σελίδας: {source_url}

Ανάλυσε το παρακάτω περιεχόμενο της ιστοσελίδας και εξήγαγε ΜΟΝΟ τους επερχόμενους αγώνες τρεξίματος (από τη σημερινή ημερομηνία και μετά).

Κανόνες:
1. Η ημερομηνία (date) πρέπει να είναι αυστηρά σε μορφή DD/MM/YYYY (π.χ. 29/11/2026). Αν δεν αναφέρεται έτος, υπολόγισε το σωστό με βάση τη σημερινή ημερομηνία ({today_str}).
2. Αν ένας αγώνας διαρκεί 2 μέρες, βάλε την ημερομηνία έναρξης.
3. Στο terrain επίλεξε αυστηρά μία από τις τιμές: "Δρόμος", "Βουνό", "Μικτό", "Στίβος".
4. Στο link βάλε το συγκεκριμένο URL του αγώνα αν υπάρχει δίπλα του, αλλιώς το {source_url}.
5. Μην επινοείς αγώνες που δεν υπάρχουν στο κείμενο.

Περιεχόμενο σελίδας:
{page_text}
"""

max_attempts = 2
    response = None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
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
            if is_temporary and attempt < max_attempts - 1:
                print("Προσωρινός φόρτος στο Gemini. Επανάληψη σε 5s...")
                time.sleep(5)
            else:
                raise

    if not response or not response.parsed:
        return []

    return [race.model_dump() for race in response.parsed.races]

def send_to_google_sheets(races: List[dict], api_url: str, api_secret: str) -> None:
    """Στέλνει τους εξαχθέντες αγώνες στο Google Apps Script Web App."""
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
    all_extracted = []

   total_sent = 0

    for url in TARGET_SOURCES:
        try:
            print(f"Σάρωση πηγής: {url}")
            text = fetch_clean_page_text(url)
            races = extract_races_with_gemini(client, url, text)
            print(f"Βρέθηκαν {len(races)} αγώνες από {url}")

            if races:
                print(f"Αποστολή {len(races)} αγώνων στο Google Sheet...")
                send_to_google_sheets(races, google_api_url, ingest_api_secret)
                total_sent += len(races)

            time.sleep(2)
        except Exception as exc:
            print(f"Σφάλμα στη σελίδα {url}: {exc}")

    print(f"Ολοκλήρωση. Συνολικοί αγώνες που στάλθηκαν για έλεγχο: {total_sent}")


if __name__ == "__main__":
    main()
