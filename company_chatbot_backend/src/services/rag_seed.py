from typing import List, Dict, Any

# This is a simple seed list for initial RAG context; in production, ingest from files or DB.
SEED_DOCS: List[Dict[str, Any]] = [
    {"text": "Our company values integrity, customer success, and innovation.", "source": "handbook"},
    {"text": "Support hours are Monday to Friday, 9am to 6pm PST.", "source": "support"},
    {"text": "Product X integrates with tools like Slack, Jira, and Salesforce.", "source": "product"},
    {"text": "For billing inquiries, contact finance@company.com.", "source": "billing"},
]
