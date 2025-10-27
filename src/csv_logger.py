"""
This module provides the AttendanceLogger class for logging recognized
individuals to a CSV file with a timestamp, with a cooldown to prevent
duplicate entries in rapid succession.
"""
import csv
import os
from datetime import datetime, timedelta

# --- Constants ---
ATTENDANCE_FILE: str = 'attendance.csv'
LOG_COOLDOWN_SECONDS: int = 60  # Cooldown in seconds before logging the same ID again

class AttendanceLogger:
    """Manages logging attendance records to a CSV file."""

    def __init__(self, filename: str = ATTENDANCE_FILE) -> None:
        """
        Initializes the logger. If the log file does not exist, it creates it
        and writes the header row.

        Args:
            filename (str): The name of the CSV file to use for logging.
        """
        self.filename: str = filename
        self.last_log_times: dict[str | int, datetime] = {}
        if not os.path.isfile(self.filename):
            try:
                with open(self.filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['SubjectID', 'Timestamp'])
                    print(f"Created new attendance file: {self.filename}")
            except IOError as e:
                print(f"Error creating attendance file: {e}")

    def log(self, subject_id: str | int) -> None:
        """
        Logs a subject's attendance if enough time has passed since their
        last log. This function handles both known integer IDs and unique
        string IDs for unknown subjects.

        Args:
            subject_id (str | int): The ID of the subject (e.g., 22) or a
                                    unique string for an unknown person.
        """
        current_time = datetime.now()
        if subject_id in self.last_log_times:
            time_since_last_log = current_time - self.last_log_times[subject_id]
            if time_since_last_log < timedelta(seconds=LOG_COOLDOWN_SECONDS):
                return

        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([subject_id, timestamp])

            self.last_log_times[subject_id] = current_time
            print(f"Logged attendance for Subject ID: {subject_id}")
        except IOError as e:
            print(f"Error writing to attendance file: {e}")