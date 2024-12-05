JSON_PATH = "result.json"
CSV_FILE_PATH = "8.csv"
REGULAR = {
    "regular1": r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$",
    "regular2": r"^[0-2]\.\d{2}$",
    "regular3": r"^\d{12}$", 
    "regular4": r"^\d{2}-\d{2}/\d{2}$",
    "regular5": r"[a-zA-Zа-яА-ЯёЁ -]+",
    "regular6": r"^-?(90(\.0+)?|[1-8]?\d(\.\d+)?)$",
    "regular7": r"^(?:AB|A|B|O)[+\u2212]$",
    "regular8": r"^\d{4}\-\d{4}$",
    "regular9": r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$",
    "regular10": r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|1\d|2[0-9]|3[0-1])$",
}