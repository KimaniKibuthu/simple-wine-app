# Create not inrange error
class NotInRange(Exception):
    def __init__(self, message="value not in range"):
        self.message = message
        super().__init__(self.message)
