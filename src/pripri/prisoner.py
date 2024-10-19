class Prisoner:
    def __init__(self, value, sensitivity):
        self._value = value
        self.sensitivity = sensitivity

    def __str__(self):
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"

    def __repr__(self):
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"
