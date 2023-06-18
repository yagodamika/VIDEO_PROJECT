import json

d = {
    "time_to_stabilize": 1,
    "time_to_binary": 2,
    "time_to_alpha": 2,
    "time_to_matted": 2,
    "time_to_output": 2,
}

d_test = json.load(open("../Outputs/timing.json", "r"))
for k in d:
    if k not in d_test:
        assert False, f"json doesnt include {k}"

print("success")

