import csv
import os


class MetricsLogger:

    FIELDNAMES = [

        "model",

        "seed",

        "episode",

        "phase",

        "opponent",

        "result",

        "reward",

        "move_count",

        "skill",

        "difficulty",

        "difficulty_name",

        "epsilon",

        "effective_epsilon",

        "q_updates_enabled",

        "q_updates",

        "learned_states",

        "avg_abs_q_change",

        "max_abs_q_change",

        "tactical_moves",

        "epsilon_moves",

        "q_policy_moves",
    ]

    def __init__(
        self,
        output_file: str
    ):

        self.output_file = output_file

        directory = os.path.dirname(
            output_file
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        self.file = open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES
        )

        self.writer.writeheader()

    def log(
        self,
        row: dict
    ):

        cleaned_row = {
            field: row.get(
                field,
                ""
            )
            for field in self.FIELDNAMES
        }

        self.writer.writerow(
            cleaned_row
        )

        # Write immediately so that progress is not
        # lost if the experiment is interrupted.
        self.file.flush()

    def close(self):

        if self.file:
            self.file.close()
            self.file = None