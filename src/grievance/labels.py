DEPARTMENTS = ["water", "roads", "power", "sanitation", "other"]
URGENCY = ["low", "medium", "high", "critical"]
SENTIMENT = ["negative", "neutral", "positive"]

# Department -> routing target (edit to match the municipal org chart)
ROUTING = {
    "water": "Water Supply & Sewerage Board",
    "roads": "Public Works Department (Roads)",
    "power": "State Electricity Distribution Company",
    "sanitation": "Municipal Solid Waste & Sanitation Dept",
    "other": "General Grievance Cell",
}

def to_id(labels, value):
    return labels.index(value)
