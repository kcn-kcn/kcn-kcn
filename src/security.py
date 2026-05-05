import hashlib

class SecurityManager:
    """Manages authentication and session security"""

    def __init__(self):
        self.valid_credentials = {
            'admin': hashlib.sha256('Admin@2025'.encode()).hexdigest(),
            'fraud_officer': hashlib.sha256('FraudDetect2025'.encode()).hexdigest(),
            'auditor': hashlib.sha256('Audit@2025'.encode()).hexdigest()
        }
        self.max_attempts = 3
        self.lockout_time = 300  # 5 minutes

    def authenticate(self, username, password):
        """Authenticate user credentials"""
        hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
        return self.valid_credentials.get(username) == hashed_pwd

    def hash_data(self, data):
        """Generate hash for data integrity"""
        return hashlib.sha256(str(data).encode()).hexdigest()
