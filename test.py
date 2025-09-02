from typing import TypedDict

class ProfileT(TypedDict):
    name: str
    age: int
    jobs: list[str]

profile: ProfileT = {
    "name": "Alice",
    "age": [30],
    "jobs": ["engineer", "teacher"]
}
