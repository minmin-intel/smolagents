import json
from typing import Dict, Tuple
import re

def parse_json_blob(json_blob: str) -> Tuple[Dict[str, str], str]:
    "Extracts the JSON blob from the input and returns the JSON data and the rest of the input."
    try:
        if "\"" not in json_blob:
            json_blob = json_blob.replace("'", '"')
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_data = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_data, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )
    

test_string = "{'tool-call': 'assistant', 'tool-response': 'user'}"
a, b = parse_json_blob(test_string)
print(a)
print(b)