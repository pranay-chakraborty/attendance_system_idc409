import csv
import os
from datetime import datetime, timedelta

ATTENDANCE_FILE = 'attendance.csv'
LOG_COOLDOWN_SECONDS = 60 

class AttendanceLogger:
    def __init__(self, filename=ATTENDANCE_FILE):
        self.filename = filename
        self.last_log_times = {}  
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['SubjectID', 'Timestamp'])

    def log(self, subject_id):
        """Logs a subject's attendance if enough time has passed since their last log."""
        current_time = datetime.now()
        
        if subject_id in self.last_log_times:
            if current_time - self.last_log_times[subject_id] < timedelta(seconds=LOG_COOLDOWN_SECONDS):
                return 
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([subject_id, timestamp])
        
        self.last_log_times[subject_id] = current_time
        print(f"Logged attendance for Subject ID: {subject_id}")