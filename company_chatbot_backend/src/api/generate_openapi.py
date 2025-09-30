import json
import os

from src.api.main import app, THEME_META

# Get the OpenAPI schema
openapi_schema = app.openapi()

# Attach Ocean Professional theme metadata at the root under 'x-style'
openapi_schema["x-style"] = {
    "applicationTheme": "Ocean Professional",
    "styleThemeData": THEME_META["colors"],
    "notes": "This API follows a modern, clean aesthetic suitable for developer-facing docs.",
}

# Write to file
output_dir = "interfaces"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "openapi.json")

with open(output_path, "w") as f:
    json.dump(openapi_schema, f, indent=2)
