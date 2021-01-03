class LoadDllError(Exception):
    def __init__(self, library: str, location: str):
        self.library = library
        self.location = location
        self.message = f"Couldn't load '{self.library}' from '{self.location}'"
