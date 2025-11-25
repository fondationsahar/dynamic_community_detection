class TimeEdge:
    def __init__(self, target, weight=1) -> None:
        # NOTE We dont store source yet
        self.target = target
        self.weight = weight
