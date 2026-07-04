"""Mock tool schemas offered to models in benchmark samples.

These are schemas only — no tool is ever executed. The benchmark judges the
tool-call objects a model emits, not tool results. All definitions use the
OpenAI ``function`` wire format; provider adapters translate as needed.
"""

from __future__ import annotations

from typing import Any


def _tool(name: str, description: str, properties: dict[str, Any],
          required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _s(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def _n(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def _i(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _b(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _arr_s(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


GET_WEATHER = _tool(
    "get_weather",
    "Get current weather for a location.",
    {
        "location": _s("City and country, e.g. 'Toronto, Canada'."),
        "unit": _s("Temperature unit.", enum=["celsius", "fahrenheit"]),
    },
    ["location", "unit"],
)

GET_WEATHER_FORECAST = _tool(
    "get_weather_forecast",
    "Get a multi-day weather forecast for a location.",
    {
        "location": _s("City name, e.g. 'Seattle'."),
        "days": _i("Number of days to forecast."),
        "unit": _s("Temperature unit.", enum=["celsius", "fahrenheit"]),
    },
    ["location", "days", "unit"],
)

GET_CURRENT_TIME = _tool(
    "get_current_time",
    "Get the current time in a timezone.",
    {"timezone": _s("IANA timezone identifier, e.g. 'Asia/Tokyo'.")},
    ["timezone"],
)

SEARCH_CALENDAR_EVENTS = _tool(
    "search_calendar_events",
    "Search calendar events by keyword. Dates are optional filters.",
    {
        "query": _s("Keyword to search event titles and descriptions."),
        "start_date": _s("Earliest date to include, format YYYY-MM-DD."),
        "end_date": _s("Latest date to include, format YYYY-MM-DD."),
    },
    ["query"],
)

CREATE_CALENDAR_EVENT = _tool(
    "create_calendar_event",
    "Create a calendar event. Dates use YYYY-MM-DD; times use 24-hour HH:MM.",
    {
        "title": _s("Event title."),
        "date": _s("Event date, format YYYY-MM-DD."),
        "start_time": _s("Start time, 24-hour format HH:MM."),
        "end_time": _s("End time, 24-hour format HH:MM."),
        "location": _s("Optional event location."),
        "attendees": _arr_s("Optional list of attendee email addresses."),
    },
    ["title", "date", "start_time", "end_time"],
)

UPDATE_CALENDAR_EVENT = _tool(
    "update_calendar_event",
    "Update fields of an existing calendar event. Only include fields being "
    "changed. Dates use YYYY-MM-DD; times use 24-hour HH:MM.",
    {
        "event_id": _s("ID of the event to update, e.g. 'EVT-1234'."),
        "changes": {
            "type": "object",
            "description": "Fields to change.",
            "properties": {
                "title": _s("New title."),
                "date": _s("New date, format YYYY-MM-DD."),
                "start_time": _s("New start time, 24-hour HH:MM."),
                "end_time": _s("New end time, 24-hour HH:MM."),
                "location": _s("New location."),
            },
            "additionalProperties": False,
        },
    },
    ["event_id", "changes"],
)

SEARCH_EMAILS = _tool(
    "search_emails",
    "Search email messages.",
    {
        "query": _s("Keyword to search subject and body."),
        "folder": _s("Folder to search.", enum=["inbox", "sent", "archive", "all"]),
        "unread_only": _b("If true, only return unread messages."),
    },
    ["query", "folder"],
)

CREATE_EMAIL_DRAFT = _tool(
    "create_email_draft",
    "Create an email draft WITHOUT sending it.",
    {
        "to": _arr_s("Recipient email addresses."),
        "subject": _s("Email subject line."),
        "body": _s("Email body text."),
        "cc": _arr_s("Optional CC email addresses."),
    },
    ["to", "subject", "body"],
)

SEND_EMAIL = _tool(
    "send_email",
    "Send an email immediately.",
    {
        "to": _arr_s("Recipient email addresses."),
        "subject": _s("Email subject line."),
        "body": _s("Email body text."),
        "cc": _arr_s("Optional CC email addresses."),
    },
    ["to", "subject", "body"],
)

LOOKUP_CONTACT = _tool(
    "lookup_contact",
    "Look up a contact record by full name.",
    {"name": _s("Full name of the contact, e.g. 'Priya Sharma'.")},
    ["name"],
)

WEB_SEARCH = _tool(
    "web_search",
    "Search the web.",
    {
        "query": _s("Search query."),
        "num_results": _i("Number of results to return."),
    },
    ["query", "num_results"],
)

CALCULATOR = _tool(
    "calculator",
    "Evaluate an arithmetic expression string, e.g. '2+2' or '847*293'.",
    {"expression": _s("Arithmetic expression to evaluate, without spaces.")},
    ["expression"],
)

CONVERT_CURRENCY = _tool(
    "convert_currency",
    "Convert an amount between currencies using ISO 4217 codes.",
    {
        "amount": _n("Amount to convert, as a number."),
        "from_currency": _s("3-letter uppercase ISO currency code, e.g. 'USD'."),
        "to_currency": _s("3-letter uppercase ISO currency code, e.g. 'EUR'."),
    },
    ["amount", "from_currency", "to_currency"],
)

CONVERT_UNITS = _tool(
    "convert_units",
    "Convert a value between measurement units. Unit names are lowercase "
    "plural words, e.g. 'miles', 'kilometers', 'fahrenheit', 'celsius'.",
    {
        "value": _n("Numeric value to convert."),
        "from_unit": _s("Source unit, lowercase."),
        "to_unit": _s("Target unit, lowercase."),
    },
    ["value", "from_unit", "to_unit"],
)

SEARCH_FILES = _tool(
    "search_files",
    "Search files in cloud storage.",
    {
        "query": _s("Keyword to search file names and contents."),
        "file_type": _s("File type filter.", enum=["pdf", "docx", "xlsx", "txt", "any"]),
        "max_results": _i("Optional maximum number of results."),
    },
    ["query", "file_type"],
)

GET_DOCUMENT = _tool(
    "get_document",
    "Retrieve a document by its exact document ID.",
    {"document_id": _s("Document ID, e.g. 'DOC-2024-0117'.")},
    ["document_id"],
)

UPDATE_SPREADSHEET_CELL = _tool(
    "update_spreadsheet_cell",
    "Set the value of a single spreadsheet cell.",
    {
        "spreadsheet_id": _s("Spreadsheet ID, e.g. 'SS-778'."),
        "sheet_name": _s("Name of the sheet tab."),
        "cell": _s("Cell reference in A1 notation, e.g. 'B2'."),
        "value": _s("New cell value, as text."),
    },
    ["spreadsheet_id", "sheet_name", "cell", "value"],
)

CREATE_GITHUB_ISSUE = _tool(
    "create_github_issue",
    "Create a GitHub issue in a repository.",
    {
        "repo": _s("Repository in 'owner/name' form, e.g. 'acme/api'."),
        "title": _s("Issue title."),
        "body": _s("Issue body text."),
        "labels": _arr_s("Optional labels to apply, in order."),
    },
    ["repo", "title", "body"],
)

GET_GITHUB_PULL_REQUEST = _tool(
    "get_github_pull_request",
    "Fetch a GitHub pull request by number.",
    {
        "repo": _s("Repository in 'owner/name' form."),
        "pr_number": _i("Pull request number."),
    },
    ["repo", "pr_number"],
)

SEND_SLACK_MESSAGE = _tool(
    "send_slack_message",
    "Post a message to a Slack channel. Channel names start with '#'.",
    {
        "channel": _s("Channel name including the leading '#', e.g. '#general'."),
        "message": _s("Message text to post."),
    },
    ["channel", "message"],
)

SEND_SLACK_DM = _tool(
    "send_slack_dm",
    "Send a direct message to a single Slack user by username (no '@' prefix).",
    {
        "user": _s("Username without the '@' prefix, e.g. 'marcus'."),
        "message": _s("Message text to send."),
    },
    ["user", "message"],
)

CREATE_REMINDER = _tool(
    "create_reminder",
    "Create a personal reminder/task. Not a calendar event — use this for "
    "simple to-dos with a due date.",
    {
        "title": _s("What to be reminded about."),
        "due_date": _s("Due date, format YYYY-MM-DD."),
        "due_time": _s("Optional due time, 24-hour HH:MM."),
        "notes": _s("Optional additional notes."),
    },
    ["title", "due_date"],
)

GEOCODE_ADDRESS = _tool(
    "geocode_address",
    "Convert a street address or place name into geographic coordinates.",
    {"address": _s("Address or place name to geocode.")},
    ["address"],
)

SEARCH_RESTAURANTS = _tool(
    "search_restaurants",
    "Find restaurants in a location.",
    {
        "location": _s("City or neighborhood, e.g. 'Lisbon, Portugal'."),
        "cuisine": _s("Optional cuisine type, lowercase, e.g. 'seafood'."),
        "price_range": _s("Optional price tier.", enum=["$", "$$", "$$$", "$$$$"]),
        "open_now": _b("If true, only restaurants open right now."),
    },
    ["location"],
)

SEARCH_FLIGHTS = _tool(
    "search_flights",
    "Search flights between airports. Airports use 3-letter uppercase IATA "
    "codes. Dates use YYYY-MM-DD.",
    {
        "origin": _s("Origin airport IATA code, e.g. 'SFO'."),
        "destination": _s("Destination airport IATA code, e.g. 'NRT'."),
        "departure_date": _s("Departure date, format YYYY-MM-DD."),
        "return_date": _s("Optional return date for round trips, YYYY-MM-DD."),
        "passengers": _i("Number of passengers."),
    },
    ["origin", "destination", "departure_date", "passengers"],
)

SEARCH_HOTELS = _tool(
    "search_hotels",
    "Search hotels in a location. Dates use YYYY-MM-DD.",
    {
        "location": _s("City and country, e.g. 'Tokyo, Japan'."),
        "check_in": _s("Check-in date, format YYYY-MM-DD."),
        "check_out": _s("Check-out date, format YYYY-MM-DD."),
        "guests": _i("Number of guests."),
    },
    ["location", "check_in", "check_out", "guests"],
)

EXECUTE_CODE = _tool(
    "execute_code",
    "Execute a code snippet and return its output.",
    {
        "language": _s("Programming language.", enum=["python", "javascript", "sql"]),
        "code": _s("Source code to execute, exactly as provided."),
    },
    ["language", "code"],
)

QUERY_DATABASE = _tool(
    "query_database",
    "Run a read-only SQL query against a named database.",
    {
        "query": _s("SQL query to execute, exactly as provided."),
        "database": _s("Database name, e.g. 'sales'."),
    },
    ["query", "database"],
)

UPDATE_CRM_CONTACT = _tool(
    "update_crm_contact",
    "Update one field of a CRM contact record.",
    {
        "contact_id": _s("CRM contact ID, e.g. 'C-4491'."),
        "field": _s("Field name to update, e.g. 'status'."),
        "value": _s("New value for the field."),
    },
    ["contact_id", "field", "value"],
)

LOOKUP_REFUND = _tool(
    "lookup_refund",
    "Look up the refund/payment status for an order.",
    {"order_id": _s("Order ID, e.g. 'ORD-88213'.")},
    ["order_id"],
)

CREATE_SUPPORT_TICKET = _tool(
    "create_support_ticket",
    "Create a customer support ticket.",
    {
        "subject": _s("Short ticket subject."),
        "description": _s("Full description of the problem."),
        "priority": _s("Ticket priority.", enum=["low", "medium", "high", "urgent"]),
        "customer_email": _s("Optional customer email address."),
    },
    ["subject", "description", "priority"],
)

ALL_TOOLS: dict[str, dict[str, Any]] = {
    t["function"]["name"]: t
    for t in (
        GET_WEATHER, GET_WEATHER_FORECAST, GET_CURRENT_TIME,
        SEARCH_CALENDAR_EVENTS, CREATE_CALENDAR_EVENT, UPDATE_CALENDAR_EVENT,
        SEARCH_EMAILS, CREATE_EMAIL_DRAFT, SEND_EMAIL, LOOKUP_CONTACT,
        WEB_SEARCH, CALCULATOR, CONVERT_CURRENCY, CONVERT_UNITS,
        SEARCH_FILES, GET_DOCUMENT, UPDATE_SPREADSHEET_CELL,
        CREATE_GITHUB_ISSUE, GET_GITHUB_PULL_REQUEST,
        SEND_SLACK_MESSAGE, SEND_SLACK_DM, CREATE_REMINDER,
        GEOCODE_ADDRESS, SEARCH_RESTAURANTS, SEARCH_FLIGHTS, SEARCH_HOTELS,
        EXECUTE_CODE, QUERY_DATABASE, UPDATE_CRM_CONTACT, LOOKUP_REFUND,
        CREATE_SUPPORT_TICKET,
    )
}
