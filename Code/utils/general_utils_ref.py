import json
import time


class Timing:
    def __init__(self, timing_output_path: str):
        self.start_time = time.time()
        self.timing_dict = {
                "time_to_stabilize": 0,
                "time_to_binary": 0,
                "time_to_alpha": 0,
                "time_to_matted": 0,
                "time_to_output": 0
            }
        self.output_path = timing_output_path

    def write_time_of_stage(self, stage: str) -> None:
        """
        Method fills time to ending of given stage from start of run
        :param stage: str of onw of the timing_dict keys
        :return: None
        """
        self.timing_dict[stage] = time.time() - self.start_time

    def write_timing_to_json(self) -> None:
        """
        Method dumps timing dict into self.output_path
        :return: None
        """
        with open(self.output_path, 'w') as json_handler:
            json.dump(self.timing_dict, json_handler, indent=4)
