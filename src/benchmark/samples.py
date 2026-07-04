"""Benchmark sample definitions, generation, loading, and validation.

The 67 samples are defined here in code. ``benchmark generate-samples``
writes them to ``benchmark_samples/tool_calling/*.json`` plus the combined
``benchmark_samples/full_benchmark.json``. The runner loads and validates
samples from the combined file (embedded copies or relative-path references).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import BENCHMARK_NAME, __version__
from .schema import Sample
from .tool_schemas import ALL_TOOLS
from .utils.hashing import sha256_of_json
from .utils.json_utils import atomic_write_json, load_json_file

EXPECTED_SAMPLE_COUNT = 67
SAMPLES_SUBDIR = "tool_calling"
FULL_BENCHMARK_FILENAME = "full_benchmark.json"

_DEFAULT_SCORING: dict[str, bool] = {
    "require_exact_tool_names": True,
    "require_exact_arguments": True,
    "allow_extra_tool_calls": False,
    "require_order": True,
    "allow_argument_coercion": False,
    "trim_string_whitespace": True,
}


def _call(name: str, /, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


def _sample(num: int, slug: str, category: str, difficulty: str, prompt: str,
            tools: list[str], expected: list[dict[str, Any]], notes: str,
            system_prompt: str = "", **scoring_overrides: bool) -> dict[str, Any]:
    scoring = dict(_DEFAULT_SCORING)
    scoring.update(scoring_overrides)
    return {
        "id": f"{num:04d}_{slug}",
        "category": category,
        "difficulty": difficulty,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "tools": [ALL_TOOLS[t] for t in tools],
        "expected_tool_calls": expected,
        "scoring": scoring,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Sample definitions
# ---------------------------------------------------------------------------

def _single_tool_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            1, "weather_current", "single_tool", "easy",
            "What is the current weather in Toronto, Canada? Use Celsius.",
            ["get_weather"],
            [_call("get_weather", location="Toronto, Canada", unit="celsius")],
            "Tests whether the model calls the weather tool with the correct "
            "location and unit.",
        ),
        _sample(
            2, "calendar_search", "single_tool", "easy",
            "Find events on my calendar that match the keyword 'dentist'. "
            "Search all dates — do not add any date filters.",
            ["search_calendar_events"],
            [_call("search_calendar_events", query="dentist")],
            "Tests a single calendar search with only the required argument; "
            "adding optional date filters fails the run.",
        ),
        _sample(
            3, "current_time_tokyo", "single_tool", "easy",
            "What time is it right now in Tokyo, Japan?",
            ["get_current_time"],
            [_call("get_current_time", timezone="Asia/Tokyo")],
            "Tests mapping a city to its IANA timezone identifier.",
        ),
        _sample(
            4, "calculator_simple", "single_tool", "easy",
            "Use the calculator to compute 847*293. Pass the expression "
            "exactly as '847*293'.",
            ["calculator"],
            [_call("calculator", expression="847*293")],
            "Tests passing an arithmetic expression through verbatim instead "
            "of answering from arithmetic ability.",
        ),
        _sample(
            5, "currency_conversion", "single_tool", "easy",
            "Convert 250 US dollars to euros.",
            ["convert_currency"],
            [_call("convert_currency", amount=250, from_currency="USD",
                   to_currency="EUR")],
            "Tests numeric amount plus ISO currency code normalization.",
        ),
        _sample(
            6, "unit_conversion", "single_tool", "easy",
            "Convert 26.2 miles to kilometers.",
            ["convert_units"],
            [_call("convert_units", value=26.2, from_unit="miles",
                   to_unit="kilometers")],
            "Tests decimal value handling and lowercase unit names.",
        ),
        _sample(
            7, "web_search_basic", "single_tool", "easy",
            "Search the web for 'latest Mars rover discoveries' and return "
            "5 results.",
            ["web_search"],
            [_call("web_search", query="latest Mars rover discoveries",
                   num_results=5)],
            "Tests a quoted query string plus an integer result count.",
        ),
        _sample(
            8, "contact_lookup", "single_tool", "easy",
            "Look up the contact record for Priya Sharma.",
            ["lookup_contact"],
            [_call("lookup_contact", name="Priya Sharma")],
            "Tests extracting a person's full name as the lookup key.",
        ),
        _sample(
            9, "email_search_unread", "single_tool", "medium",
            "Search my inbox for unread emails matching 'invoice'.",
            ["search_emails"],
            [_call("search_emails", query="invoice", folder="inbox",
                   unread_only=True)],
            "Tests mapping 'inbox' and 'unread' phrasing onto an enum folder "
            "and a boolean flag.",
        ),
        _sample(
            10, "file_search_pdf", "single_tool", "medium",
            "Search my files for PDFs matching 'quarterly report'. Limit to "
            "10 results.",
            ["search_files"],
            [_call("search_files", query="quarterly report", file_type="pdf",
                   max_results=10)],
            "Tests an enum file type plus an optional integer limit that the "
            "prompt makes required.",
        ),
        _sample(
            11, "document_retrieval", "single_tool", "easy",
            "Fetch the document with ID DOC-2024-0117.",
            ["get_document"],
            [_call("get_document", document_id="DOC-2024-0117")],
            "Tests copying an exact document ID.",
        ),
        _sample(
            12, "github_pr_lookup", "single_tool", "easy",
            "Show me pull request #482 in the acme/website repository.",
            ["get_github_pull_request"],
            [_call("get_github_pull_request", repo="acme/website",
                   pr_number=482)],
            "Tests an owner/name repo string and an integer PR number "
            "(not the string '#482').",
        ),
        _sample(
            13, "slack_message", "single_tool", "easy",
            'Post exactly "Deploy complete." to the #deployments Slack '
            "channel.",
            ["send_slack_message"],
            [_call("send_slack_message", channel="#deployments",
                   message="Deploy complete.")],
            "Tests verbatim message text and a channel name with its '#'.",
        ),
        _sample(
            14, "reminder_creation", "single_tool", "easy",
            "Create a reminder titled 'Renew passport' due on 2026-08-15.",
            ["create_reminder"],
            [_call("create_reminder", title="Renew passport",
                   due_date="2026-08-15")],
            "Tests a quoted title and an ISO due date with no extra "
            "optional arguments.",
        ),
        _sample(
            15, "geocode_address", "single_tool", "easy",
            "Geocode the address '1600 Pennsylvania Avenue NW, Washington, "
            "DC'.",
            ["geocode_address"],
            [_call("geocode_address",
                   address="1600 Pennsylvania Avenue NW, Washington, DC")],
            "Tests passing a multi-part street address through verbatim.",
        ),
        _sample(
            16, "restaurant_search", "single_tool", "medium",
            "Find seafood restaurants in Lisbon, Portugal that are open "
            "right now.",
            ["search_restaurants"],
            [_call("search_restaurants", location="Lisbon, Portugal",
                   cuisine="seafood", open_now=True)],
            "Tests selecting the right optional arguments (cuisine, "
            "open_now) while leaving price_range out.",
        ),
        _sample(
            17, "database_query", "single_tool", "medium",
            "Run this SQL against the 'sales' database: "
            "SELECT COUNT(*) FROM orders WHERE status = 'pending'",
            ["query_database"],
            [_call("query_database",
                   query="SELECT COUNT(*) FROM orders WHERE status = 'pending'",
                   database="sales")],
            "Tests passing a SQL string through without rewriting it.",
        ),
        _sample(
            18, "code_execution", "single_tool", "medium",
            "Run this Python code exactly as written: "
            "print(sum(range(1, 101)))",
            ["execute_code"],
            [_call("execute_code", language="python",
                   code="print(sum(range(1, 101)))")],
            "Tests a language enum plus verbatim code, resisting the "
            "temptation to compute the answer directly.",
        ),
        _sample(
            19, "refund_lookup", "single_tool", "easy",
            "Check the refund status for order ORD-88213.",
            ["lookup_refund"],
            [_call("lookup_refund", order_id="ORD-88213")],
            "Tests copying an exact order ID.",
        ),
        _sample(
            20, "hotel_search", "single_tool", "medium",
            "Find hotels in Barcelona, Spain for 2 guests, checking in "
            "2026-09-04 and out 2026-09-08.",
            ["search_hotels"],
            [_call("search_hotels", location="Barcelona, Spain",
                   check_in="2026-09-04", check_out="2026-09-08", guests=2)],
            "Tests four required arguments including two ISO dates and an "
            "integer guest count.",
        ),
    ]


def _multi_tool_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            21, "event_and_reminder", "multi_tool", "medium",
            "Two tasks: (1) Create a calendar event titled 'Team offsite' on "
            "2026-09-12 from 09:00 to 17:00. (2) Create a reminder titled "
            "'Book offsite venue' due on 2026-08-29.",
            ["create_calendar_event", "create_reminder"],
            [
                _call("create_calendar_event", title="Team offsite",
                      date="2026-09-12", start_time="09:00", end_time="17:00"),
                _call("create_reminder", title="Book offsite venue",
                      due_date="2026-08-29"),
            ],
            "Two different tools must both be called; order does not matter.",
            require_order=False,
        ),
        _sample(
            22, "email_search_and_draft", "multi_tool", "medium",
            "Search all my email folders (folder 'all') for messages "
            "matching 'contract renewal'. Also draft — but do not send — an "
            "email to legal@example.com with subject 'Contract renewal "
            "status' and body 'Please send an update on the renewal.'",
            ["search_emails", "create_email_draft", "send_email"],
            [
                _call("search_emails", query="contract renewal", folder="all"),
                _call("create_email_draft", to=["legal@example.com"],
                      subject="Contract renewal status",
                      body="Please send an update on the renewal."),
            ],
            "Requires a search plus a draft; calling send_email instead of "
            "create_email_draft fails the run.",
            require_order=False,
        ),
        _sample(
            23, "github_issue_and_slack", "multi_tool", "medium",
            "File a GitHub issue in acme/api titled 'Rate limiter returns "
            "500 under load' with body 'Load tests above 200 rps trigger "
            "HTTP 500 responses from the rate limiter.' and the single "
            'label \'bug\'. Also post exactly "Filed the rate limiter '
            'issue." to the #eng-alerts channel.',
            ["create_github_issue", "send_slack_message"],
            [
                _call("create_github_issue", repo="acme/api",
                      title="Rate limiter returns 500 under load",
                      body="Load tests above 200 rps trigger HTTP 500 "
                           "responses from the rate limiter.",
                      labels=["bug"]),
                _call("send_slack_message", channel="#eng-alerts",
                      message="Filed the rate limiter issue."),
            ],
            "An issue plus a notification; both quoted texts must be "
            "verbatim.",
            require_order=False,
        ),
        _sample(
            24, "crm_and_ticket", "multi_tool", "medium",
            "Update CRM contact C-4491: set the field 'status' to the value "
            "'churn_risk'. Also open a support ticket with subject "
            "'Repeated login failures', description 'Customer C-4491 "
            "reports login failures since the last release.', and high "
            "priority.",
            ["update_crm_contact", "create_support_ticket"],
            [
                _call("update_crm_contact", contact_id="C-4491",
                      field="status", value="churn_risk"),
                _call("create_support_ticket",
                      subject="Repeated login failures",
                      description="Customer C-4491 reports login failures "
                                  "since the last release.",
                      priority="high"),
            ],
            "A CRM field update plus ticket creation with an enum priority.",
            require_order=False,
        ),
        _sample(
            25, "flight_and_hotel", "multi_tool", "hard",
            "Plan my Tokyo trip: search round-trip flights for 2 passengers "
            "from SFO to NRT departing 2026-10-02 and returning 2026-10-16, "
            "and search hotels in Tokyo, Japan for 2 guests checking in "
            "2026-10-02 and out 2026-10-16.",
            ["search_flights", "search_hotels"],
            [
                _call("search_flights", origin="SFO", destination="NRT",
                      departure_date="2026-10-02", return_date="2026-10-16",
                      passengers=2),
                _call("search_hotels", location="Tokyo, Japan",
                      check_in="2026-10-02", check_out="2026-10-16",
                      guests=2),
            ],
            "Nine precise arguments across two travel tools.",
            require_order=False,
        ),
        _sample(
            26, "geocode_and_weather", "multi_tool", "medium",
            "Geocode the address '221B Baker Street, London' and get the "
            "current weather in 'London, UK' in celsius.",
            ["geocode_address", "get_weather"],
            [
                _call("geocode_address", address="221B Baker Street, London"),
                _call("get_weather", location="London, UK", unit="celsius"),
            ],
            "Two independent location tools with distinct location strings.",
            require_order=False,
        ),
        _sample(
            27, "db_and_spreadsheet", "multi_tool", "hard",
            "Run this SQL against the 'analytics' database: SELECT region, "
            "SUM(revenue) FROM sales_2025 GROUP BY region — and set cell B2 "
            "of the 'Summary' sheet in spreadsheet SS-778 to the text "
            "'refresh scheduled'.",
            ["query_database", "update_spreadsheet_cell"],
            [
                _call("query_database",
                      query="SELECT region, SUM(revenue) FROM sales_2025 "
                            "GROUP BY region",
                      database="analytics"),
                _call("update_spreadsheet_cell", spreadsheet_id="SS-778",
                      sheet_name="Summary", cell="B2",
                      value="refresh scheduled"),
            ],
            "A verbatim SQL string plus a four-argument spreadsheet update.",
            require_order=False,
        ),
        _sample(
            28, "time_and_event", "multi_tool", "medium",
            "Check the current time in the 'Europe/Paris' timezone, and "
            "create a calendar event titled 'Paris standup' on 2026-07-10 "
            "from 09:30 to 09:45.",
            ["get_current_time", "create_calendar_event"],
            [
                _call("get_current_time", timezone="Europe/Paris"),
                _call("create_calendar_event", title="Paris standup",
                      date="2026-07-10", start_time="09:30",
                      end_time="09:45"),
            ],
            "A read-only lookup plus an event creation in one turn.",
            require_order=False,
        ),
        _sample(
            29, "files_and_document", "multi_tool", "medium",
            "Search my files for Word documents (docx) matching 'onboarding "
            "checklist', and also fetch the document with ID "
            "DOC-ONBOARD-01.",
            ["search_files", "get_document"],
            [
                _call("search_files", query="onboarding checklist",
                      file_type="docx"),
                _call("get_document", document_id="DOC-ONBOARD-01"),
            ],
            "A filtered file search plus a direct document fetch.",
            require_order=False,
        ),
    ]

def _parallel_tools_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            30, "weather_three_cities", "parallel_tools", "easy",
            "Get the current weather, in celsius, for each of these three "
            "cities: 'Toronto, Canada', 'Berlin, Germany', and "
            "'Sydney, Australia'. Make one weather call per city.",
            ["get_weather"],
            [
                _call("get_weather", location="Toronto, Canada", unit="celsius"),
                _call("get_weather", location="Berlin, Germany", unit="celsius"),
                _call("get_weather", location="Sydney, Australia", unit="celsius"),
            ],
            "Three independent calls to the same tool, any order.",
            require_order=False,
        ),
        _sample(
            31, "currency_three_targets", "parallel_tools", "medium",
            "Convert 100 USD into each of EUR, GBP, and JPY — one "
            "conversion call per target currency.",
            ["convert_currency"],
            [
                _call("convert_currency", amount=100, from_currency="USD",
                      to_currency="EUR"),
                _call("convert_currency", amount=100, from_currency="USD",
                      to_currency="GBP"),
                _call("convert_currency", amount=100, from_currency="USD",
                      to_currency="JPY"),
            ],
            "Fan-out over target currencies with a shared amount and source.",
            require_order=False,
        ),
        _sample(
            32, "time_three_zones", "parallel_tools", "easy",
            "Get the current time in each of these timezones: "
            "'America/New_York', 'Europe/London', and 'Asia/Singapore'.",
            ["get_current_time"],
            [
                _call("get_current_time", timezone="America/New_York"),
                _call("get_current_time", timezone="Europe/London"),
                _call("get_current_time", timezone="Asia/Singapore"),
            ],
            "Three parallel time lookups with explicit IANA zones.",
            require_order=False,
        ),
        _sample(
            33, "geocode_two_addresses", "parallel_tools", "medium",
            "Geocode both of these addresses, one call each: 'Piazza San "
            "Marco, Venice, Italy' and 'Alexanderplatz, Berlin, Germany'.",
            ["geocode_address"],
            [
                _call("geocode_address",
                      address="Piazza San Marco, Venice, Italy"),
                _call("geocode_address",
                      address="Alexanderplatz, Berlin, Germany"),
            ],
            "Two verbatim addresses, one call per address.",
            require_order=False,
        ),
        _sample(
            34, "web_search_two_topics", "parallel_tools", "medium",
            "Run two web searches with 3 results each: one for 'electric "
            "vehicle battery recycling' and one for 'solid state battery "
            "breakthrough'.",
            ["web_search"],
            [
                _call("web_search", query="electric vehicle battery recycling",
                      num_results=3),
                _call("web_search", query="solid state battery breakthrough",
                      num_results=3),
            ],
            "Two searches sharing a result count; queries must be verbatim.",
            require_order=False,
        ),
        _sample(
            35, "github_three_prs", "parallel_tools", "medium",
            "Fetch pull requests #101, #102, and #103 from the acme/mobile "
            "repository — one call per pull request.",
            ["get_github_pull_request"],
            [
                _call("get_github_pull_request", repo="acme/mobile", pr_number=101),
                _call("get_github_pull_request", repo="acme/mobile", pr_number=102),
                _call("get_github_pull_request", repo="acme/mobile", pr_number=103),
            ],
            "Batch lookup of three PR numbers as integers.",
            require_order=False,
        ),
        _sample(
            36, "flights_two_routes", "parallel_tools", "hard",
            "Compare one-way options for 1 passenger on 2026-11-20: search "
            "JFK to LHR, and separately search EWR to LHR. Two searches, no "
            "return date.",
            ["search_flights"],
            [
                _call("search_flights", origin="JFK", destination="LHR",
                      departure_date="2026-11-20", passengers=1),
                _call("search_flights", origin="EWR", destination="LHR",
                      departure_date="2026-11-20", passengers=1),
            ],
            "Two one-way searches; adding a return_date is an argument "
            "error.",
            require_order=False,
        ),
        _sample(
            37, "refunds_batch", "parallel_tools", "medium",
            "Check the refund status for each of these orders: ORD-10001, "
            "ORD-10002, and ORD-10003.",
            ["lookup_refund"],
            [
                _call("lookup_refund", order_id="ORD-10001"),
                _call("lookup_refund", order_id="ORD-10002"),
                _call("lookup_refund", order_id="ORD-10003"),
            ],
            "One lookup per order ID.",
            require_order=False,
        ),
    ]


def _no_tool_decoy_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            38, "general_knowledge_capital", "no_tool_decoy", "easy",
            "What is the capital of Australia?",
            ["web_search"],
            [],
            "Stable general knowledge; searching the web is an unnecessary "
            "tool call and fails the run.",
        ),
        _sample(
            39, "trivial_arithmetic", "no_tool_decoy", "medium",
            "What is 2 + 2?",
            ["calculator"],
            [],
            "Trivial arithmetic with a calculator dangled as bait.",
        ),
        _sample(
            40, "gratitude_message", "no_tool_decoy", "easy",
            "Thanks so much for your help earlier — everything works now!",
            ["send_slack_message", "send_email"],
            [],
            "Pure conversational gratitude; no action was requested.",
        ),
        _sample(
            41, "opinion_question", "no_tool_decoy", "easy",
            "In your opinion, what makes code easy to read?",
            ["web_search", "query_database"],
            [],
            "An opinion question that neither search nor a database can "
            "answer.",
        ),
        _sample(
            42, "info_already_given", "no_tool_decoy", "medium",
            "It's currently 22°C and sunny in Madrid. Should I bring a "
            "jacket for an evening walk?",
            ["get_weather"],
            [],
            "The weather is already stated in the prompt; calling the "
            "weather tool anyway fails the run.",
        ),
        _sample(
            43, "hypothetical_flight", "no_tool_decoy", "medium",
            "Roughly how long is a nonstop flight from New York to London? "
            "No need to search — a ballpark figure is fine.",
            ["search_flights"],
            [],
            "The user explicitly declines a search; a flight search anyway "
            "fails the run.",
        ),
        _sample(
            44, "acronym_definition", "no_tool_decoy", "easy",
            "What does the acronym 'API' stand for?",
            ["web_search", "get_document"],
            [],
            "A definition every model knows; both offered tools are decoys.",
        ),
        _sample(
            45, "creative_writing", "no_tool_decoy", "medium",
            "Write me a haiku about autumn leaves.",
            ["execute_code", "calculator"],
            [],
            "Creative writing with irrelevant technical tools offered.",
        ),
    ]


def _argument_precision_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            46, "email_exact_recipients", "argument_precision", "medium",
            "Send an email to alice.chen@example.com and "
            "raj.patel@example.com with subject 'Q3 planning follow-up' and "
            "body 'Slides are attached in the shared drive.'",
            ["send_email"],
            [_call("send_email",
                   to=["alice.chen@example.com", "raj.patel@example.com"],
                   subject="Q3 planning follow-up",
                   body="Slides are attached in the shared drive.")],
            "Both recipients must appear, in the order given, with verbatim "
            "subject and body.",
        ),
        _sample(
            47, "event_datetime_conversion", "argument_precision", "hard",
            "Book a calendar event titled 'Dentist appointment' on November "
            "3, 2026 from 2:30pm to 3:15pm.",
            ["create_calendar_event"],
            [_call("create_calendar_event", title="Dentist appointment",
                   date="2026-11-03", start_time="14:30", end_time="15:15")],
            "Natural-language date and 12-hour times must be converted to "
            "the YYYY-MM-DD and 24-hour HH:MM formats the schema requires.",
        ),
        _sample(
            48, "spreadsheet_text_value", "argument_precision", "medium",
            "In spreadsheet SS-2026-BUDGET, on the 'Q4' sheet, set cell C17 "
            "to the text value '48250'.",
            ["update_spreadsheet_cell"],
            [_call("update_spreadsheet_cell", spreadsheet_id="SS-2026-BUDGET",
                   sheet_name="Q4", cell="C17", value="48250")],
            "The value must be the string '48250', not the number 48250 — a "
            "JSON type trap.",
        ),
        _sample(
            49, "ticket_priority_enum", "argument_precision", "medium",
            "Open an urgent support ticket for sam.lee@example.com. "
            "Subject: 'Cannot access account'. Description: 'User reports "
            "2FA codes are never delivered.'",
            ["create_support_ticket"],
            [_call("create_support_ticket", subject="Cannot access account",
                   description="User reports 2FA codes are never delivered.",
                   priority="urgent", customer_email="sam.lee@example.com")],
            "The word 'urgent' must map to the exact enum value, and the "
            "optional customer_email must be included.",
        ),
        _sample(
            50, "currency_decimal_amount", "argument_precision", "medium",
            "Convert 1,499.99 Canadian dollars to US dollars.",
            ["convert_currency"],
            [_call("convert_currency", amount=1499.99, from_currency="CAD",
                   to_currency="USD")],
            "A thousands separator must not survive into the numeric "
            "amount; currency names map to ISO codes.",
        ),
        _sample(
            51, "github_labels_order", "argument_precision", "hard",
            "Create an issue in acme/mobile-app titled 'Crash on launch "
            "when offline' with body 'App crashes on cold start when the "
            "device has no connectivity.' Apply these labels in this exact "
            "order: bug, ios, p1.",
            ["create_github_issue"],
            [_call("create_github_issue", repo="acme/mobile-app",
                   title="Crash on launch when offline",
                   body="App crashes on cold start when the device has no "
                        "connectivity.",
                   labels=["bug", "ios", "p1"])],
            "An ordered array argument: the labels list must match "
            "element-for-element in order.",
        ),
        _sample(
            52, "flight_iata_codes", "argument_precision", "hard",
            "Search flights for just me from Toronto Pearson to Paris "
            "Charles de Gaulle, departing 2026-12-19 and returning "
            "2027-01-04. Use IATA airport codes.",
            ["search_flights"],
            [_call("search_flights", origin="YYZ", destination="CDG",
                   departure_date="2026-12-19", return_date="2027-01-04",
                   passengers=1)],
            "Airport names must become IATA codes and 'just me' must become "
            "passengers=1.",
        ),
        _sample(
            53, "unit_exact_names", "argument_precision", "medium",
            "Convert 98.6 degrees Fahrenheit to Celsius.",
            ["convert_units"],
            [_call("convert_units", value=98.6, from_unit="fahrenheit",
                   to_unit="celsius")],
            "Unit names must be lowercased exactly as the schema describes.",
        ),
        _sample(
            54, "sql_exact_string", "argument_precision", "hard",
            "Run exactly this query against the 'customers_prod' database: "
            "SELECT id, email FROM customers WHERE created_at >= "
            "'2026-01-01' ORDER BY created_at DESC LIMIT 50",
            ["query_database"],
            [_call("query_database",
                   query="SELECT id, email FROM customers WHERE created_at "
                         ">= '2026-01-01' ORDER BY created_at DESC LIMIT 50",
                   database="customers_prod")],
            "A long SQL string with embedded quotes must pass through "
            "byte-for-byte.",
        ),
        _sample(
            55, "nested_event_update", "argument_precision", "hard",
            "Move event EVT-3391 to 2026-08-07 with a new start time of "
            "10:00. Change nothing else.",
            ["update_calendar_event"],
            [_call("update_calendar_event", event_id="EVT-3391",
                   changes={"date": "2026-08-07", "start_time": "10:00"})],
            "A nested object argument must contain exactly two keys — "
            "adding or omitting fields inside 'changes' fails the run.",
        ),
    ]


def _ordering_required_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            56, "lookup_then_notify", "ordering_required", "medium",
            "First look up the contact record for Ana Silva. Then post "
            'exactly "Reached out to Ana" in the #sales channel. Do it in '
            "that order.",
            ["lookup_contact", "send_slack_message"],
            [
                _call("lookup_contact", name="Ana Silva"),
                _call("send_slack_message", channel="#sales",
                      message="Reached out to Ana"),
            ],
            "The lookup must come before the notification.",
            require_order=True,
        ),
        _sample(
            57, "query_then_record", "ordering_required", "hard",
            "In this exact order: first run the query SELECT COUNT(*) FROM "
            "signups WHERE plan = 'pro' against the 'growth' database, then "
            "set cell D4 on the 'Metrics' sheet of spreadsheet "
            "SS-METRICS-26 to the text 'exported'.",
            ["query_database", "update_spreadsheet_cell"],
            [
                _call("query_database",
                      query="SELECT COUNT(*) FROM signups WHERE plan = 'pro'",
                      database="growth"),
                _call("update_spreadsheet_cell",
                      spreadsheet_id="SS-METRICS-26", sheet_name="Metrics",
                      cell="D4", value="exported"),
            ],
            "Export happens only after the query; reversed order fails.",
            require_order=True,
        ),
        _sample(
            58, "geocode_then_restaurants", "ordering_required", "medium",
            "First geocode 'Shibuya Crossing, Tokyo', then search for ramen "
            "restaurants in 'Shibuya, Tokyo'. Keep that order.",
            ["geocode_address", "search_restaurants"],
            [
                _call("geocode_address", address="Shibuya Crossing, Tokyo"),
                _call("search_restaurants", location="Shibuya, Tokyo",
                      cuisine="ramen"),
            ],
            "Locate first, then search nearby restaurants.",
            require_order=True,
        ),
        _sample(
            59, "time_then_reminder", "ordering_required", "medium",
            "First check the current time in 'America/Los_Angeles', then "
            "create a reminder titled 'Submit expense report' due "
            "2026-07-31. In that order.",
            ["get_current_time", "create_reminder"],
            [
                _call("get_current_time", timezone="America/Los_Angeles"),
                _call("create_reminder", title="Submit expense report",
                      due_date="2026-07-31"),
            ],
            "A read-then-write sequence that must not be reordered.",
            require_order=True,
        ),
        _sample(
            60, "search_then_fetch", "ordering_required", "medium",
            "First search my files for PDFs matching 'security audit', then "
            "fetch the document DOC-SEC-2026. The search must come first.",
            ["search_files", "get_document"],
            [
                _call("search_files", query="security audit", file_type="pdf"),
                _call("get_document", document_id="DOC-SEC-2026"),
            ],
            "Search precedes retrieval.",
            require_order=True,
        ),
        _sample(
            61, "ticket_then_email", "ordering_required", "hard",
            "In this exact order: first create a high-priority support "
            "ticket with subject 'Checkout latency spike' and description "
            "'p95 checkout latency exceeded 4 seconds after the 14:00 "
            "deploy.', then send an email to ops@example.com with subject "
            "'Ticket filed: checkout latency' and body 'Tracking the "
            "latency spike in a new high-priority ticket.'",
            ["create_support_ticket", "send_email"],
            [
                _call("create_support_ticket",
                      subject="Checkout latency spike",
                      description="p95 checkout latency exceeded 4 seconds "
                                  "after the 14:00 deploy.",
                      priority="high"),
                _call("send_email", to=["ops@example.com"],
                      subject="Ticket filed: checkout latency",
                      body="Tracking the latency spike in a new "
                           "high-priority ticket."),
            ],
            "Ticket first, notification second, with verbatim text in both.",
            require_order=True,
        ),
    ]


def _ambiguity_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            62, "weather_now_vs_forecast", "tool_choice_under_ambiguity",
            "medium",
            "Will it rain in Seattle over the next 5 days? Use celsius.",
            ["get_weather", "get_weather_forecast"],
            [_call("get_weather_forecast", location="Seattle", days=5,
                   unit="celsius")],
            "'Next 5 days' requires the forecast tool, not current weather.",
        ),
        _sample(
            63, "draft_vs_send", "tool_choice_under_ambiguity", "medium",
            "Prepare an email to jordan@example.com with subject 'Renewal "
            "quote' and body 'Quote attached for the annual renewal.' — but "
            "do NOT send it yet; I want to review it first.",
            ["create_email_draft", "send_email"],
            [_call("create_email_draft", to=["jordan@example.com"],
                   subject="Renewal quote",
                   body="Quote attached for the annual renewal.")],
            "'Do not send' means the draft tool; send_email fails the run.",
        ),
        _sample(
            64, "reminder_vs_event", "tool_choice_under_ambiguity", "medium",
            "Add a to-do for me: remind me to water the plants on "
            "2026-07-10. Title it 'Water the plants'.",
            ["create_reminder", "create_calendar_event"],
            [_call("create_reminder", title="Water the plants",
                   due_date="2026-07-10")],
            "A dated to-do is a reminder, not a calendar event.",
        ),
        _sample(
            65, "channel_vs_dm", "tool_choice_under_ambiguity", "medium",
            "Send a direct message to the Slack user marcus saying exactly "
            '"Standup moved to 10am."',
            ["send_slack_message", "send_slack_dm"],
            [_call("send_slack_dm", user="marcus",
                   message="Standup moved to 10am.")],
            "A DM to a user must use send_slack_dm, whose schema takes a "
            "bare username, not a '#channel'.",
        ),
        _sample(
            66, "db_vs_calc_vs_search", "tool_choice_under_ambiguity", "hard",
            "How many orders did we get in June 2026? Our own 'sales' "
            "database has the answer — run exactly: SELECT COUNT(*) FROM "
            "orders WHERE order_date BETWEEN '2026-06-01' AND '2026-06-30'. "
            "Don't search the web and don't try to compute it yourself.",
            ["query_database", "calculator", "web_search"],
            [_call("query_database",
                   query="SELECT COUNT(*) FROM orders WHERE order_date "
                         "BETWEEN '2026-06-01' AND '2026-06-30'",
                   database="sales")],
            "Three plausible tools; only the database query is correct.",
        ),
    ]


def _final_workflow_sample() -> list[dict[str, Any]]:
    return [
        _sample(
            67, "multi_tool_complex_workflow", "multi_tool", "hard",
            "Set up my contract review: (1) look up the contact record for "
            "Miguel Ortiz; (2) create a calendar event titled 'Contract "
            "review with Miguel Ortiz' on 2026-07-21 from 14:00 to 15:00; "
            "(3) send an email to miguel.ortiz@example.com with subject "
            "'Contract review on July 21' and body 'Confirming our contract "
            "review at 2pm on July 21.'; (4) create a reminder titled "
            "'Prepare contract notes' due on 2026-07-20.",
            ["lookup_contact", "create_calendar_event", "send_email",
             "create_reminder"],
            [
                _call("lookup_contact", name="Miguel Ortiz"),
                _call("create_calendar_event",
                      title="Contract review with Miguel Ortiz",
                      date="2026-07-21", start_time="14:00",
                      end_time="15:00"),
                _call("send_email", to=["miguel.ortiz@example.com"],
                      subject="Contract review on July 21",
                      body="Confirming our contract review at 2pm on "
                           "July 21."),
                _call("create_reminder", title="Prepare contract notes",
                      due_date="2026-07-20"),
            ],
            "Four tools in one workflow; every argument is pinned by the "
            "prompt, order is not enforced.",
            require_order=False,
        ),
    ]


def build_all_samples() -> list[dict[str, Any]]:
    """All 67 samples as plain dicts, sorted by id."""
    samples = (
        _single_tool_samples()
        + _multi_tool_samples()
        + _parallel_tools_samples()
        + _no_tool_decoy_samples()
        + _argument_precision_samples()
        + _ordering_required_samples()
        + _ambiguity_samples()
        + _final_workflow_sample()
    )
    samples.sort(key=lambda s: s["id"])
    return samples


# ---------------------------------------------------------------------------
# Validation / generation / loading
# ---------------------------------------------------------------------------

class SampleValidationError(ValueError):
    """Raised when any sample fails schema validation."""


def validate_samples(raw_samples: list[dict[str, Any]]) -> list[Sample]:
    """Validate raw sample dicts. Raises before any API call on failure."""
    errors: list[str] = []
    validated: list[Sample] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_samples):
        label = raw.get("id", f"<index {i}>") if isinstance(raw, dict) else f"<index {i}>"
        try:
            sample = Sample.model_validate(raw)
            sample.validate_expected_calls_reference_known_tools()
        except Exception as exc:  # pydantic ValidationError or ValueError
            errors.append(f"sample {label}: {exc}")
            continue
        if sample.id in seen_ids:
            errors.append(f"sample {sample.id}: duplicate id")
            continue
        seen_ids.add(sample.id)
        validated.append(sample)
    if errors:
        raise SampleValidationError(
            "sample validation failed:\n  - " + "\n  - ".join(errors)
        )
    validated.sort(key=lambda s: s.id)
    return validated


def generate_sample_files(base_dir: str | Path = "benchmark_samples") -> Path:
    """Write the 67 per-sample JSON files and full_benchmark.json.

    Returns the path to the combined benchmark file.
    """
    base = Path(base_dir)
    samples = build_all_samples()
    validate_samples(samples)  # never write invalid samples
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise SampleValidationError(
            f"expected {EXPECTED_SAMPLE_COUNT} samples, built {len(samples)}"
        )
    sample_dir = base / SAMPLES_SUBDIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        atomic_write_json(sample_dir / f"{sample['id']}.json", sample)
    combined = {
        "benchmark": BENCHMARK_NAME,
        "version": __version__,
        "sample_count": len(samples),
        "samples": samples,
    }
    combined_path = base / FULL_BENCHMARK_FILENAME
    atomic_write_json(combined_path, combined)
    return combined_path


def load_benchmark_file(path: str | Path) -> list[Sample]:
    """Load and validate samples from a combined benchmark file.

    Accepts either embedded sample objects or relative-path string references
    in the ``samples`` array (resolved against the benchmark file's directory).
    A bare top-level array of samples is also accepted.
    """
    path = Path(path)
    data = load_json_file(path)
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        raw_list = data["samples"]
    else:
        raise SampleValidationError(
            f"{path}: expected an object with a 'samples' array "
            f"(or a bare array of samples)"
        )
    resolved: list[dict[str, Any]] = []
    for entry in raw_list:
        if isinstance(entry, str):
            ref = (path.parent / entry).resolve()
            resolved.append(load_json_file(ref))
        elif isinstance(entry, dict):
            resolved.append(entry)
        else:
            raise SampleValidationError(
                f"{path}: sample entries must be objects or path strings, "
                f"got {type(entry).__name__}"
            )
    return validate_samples(resolved)


def samples_hash(samples: list[Sample]) -> str:
    """Deterministic hash over the validated, id-sorted sample set."""
    payload = [s.model_dump(mode="json") for s in
               sorted(samples, key=lambda s: s.id)]
    return sha256_of_json(payload)

