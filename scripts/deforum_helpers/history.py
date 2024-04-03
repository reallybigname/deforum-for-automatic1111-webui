import copy
import uuid
from collections import OrderedDict

class ThingHistory:
    """A class to store a history of keys/states."""

    def __init__(self, max_states=0):
        """Initialize ThingHistory with max_states (set to 0 and will store nothing unless max_states is provided)."""
        self.history = OrderedDict()
        self.max_states = max_states

    @property
    def length(self):
        """Get current count of history states with ThingHistory.length."""
        return len(self.history)

    def list_keys(self):
        """Get a list of all keys in the history with ThingHistory.list_keys()"""
        return list(self.history.keys())

    def list_states(self):
        """Get a list of all states in the history with ThingHistory.list_states()"""
        return list(self.history.values())

    def add_state(self, state, state_key=None, overwrite=True):
        """Inserts a state in ThingHistory with an optional key. Removes last state if at max_states."""
        if self.max_states > 0:

            # if state_key is provided
            if state_key is not None:
                # Convert state_key to a string if it isn't one
                state_key = self._convert_state_key_to_string(state_key)
                # Validate that state_key doesn't already exist
                if state_key in self.history and not overwrite:
                    raise ValueError(f"State key {state_key} already exists in the history.")

            keys = self.list_keys()

            # If no state_key provided
            if state_key is None:
                # check that last key is digit and is a non-negative integer
                if keys and keys[0].isdigit() and int(keys[0]) >= 0:
                    # use the next integer as state key
                    state_key = str(int(keys[0]) + 1)
                else:
                    # If last key is not a non-negative integer, generate a UUID
                    state_key = str(uuid.uuid4())

            new_history = OrderedDict()
            new_history[state_key] = copy.deepcopy(state) # Use deepcopy to ensure a new object is created
            self.history = OrderedDict(list(new_history.items()) + list(self.history.items()))
            self._regulate()

    def __getitem__(self, nth):
        """Get nth state with ThingHistory[nth]"""
        self._validate_nth(nth) # Use the helper method to validate nth
        keys = self.list_keys()
        return copy.deepcopy(self.history[keys[nth-1]]) # Use deepcopy to return a new object

    def pop(self, nth=None):
        """Pops and returns the nth key/state in a tuple."""
        """Overrides normal pop() behavior of dict. Will pop from oldest history item if no nth param provided."""
        keys = self.list_keys()
        nth = self.length if nth is None else self._validate_nth(nth)
        # get state_key for nth, pop
        state_key = keys[nth-1]
        state = self.history.pop(state_key)
        # return key as int if it matches int
        state_key = self.int_str_to_int(state_key)
        return state, state_key # don't need deepcopy because the key has been removed anyway

    def delete_nth(self, nth):
        """Deletes the nth key/state if it exists."""
        self._validate_nth(nth) # Use the helper method to validate nth
        keys = self.list_keys()
        self.history.pop(keys[nth-1])

    def get_nth(self, nth=None):
        """Return nth key and state in a tuple. (raises ValueError if no nth available)."""
        """Will get oldest nth in history if no nth param provided"""
        keys = self.list_keys()
        nth = self.length if nth is None else self._validate_nth(nth)
        return self.__getitem__(nth), keys[nth-1] # No need to use deepcopy here

    def get_nth_key(self, nth=None):
        """Return the key of the nth state. (raises ValueError if no nth available)."""
        """Will get oldest nth in history if no nth param provided"""
        keys = self.list_keys()
        nth = self.length if nth is None else self._validate_nth(nth)
        # return state_key for nth
        return self.int_str_to_int(keys[nth-1])

    def get_by_key(self, state_key):
        """Get state by key."""
        self._validate_state_key_not_none(state_key)
        state_key = self._convert_state_key_to_string(state_key)
        self._validate_state_key_exists(state_key) # Use the helper method to validate state key exists
        return copy.deepcopy(self.history[state_key]) # Use deepcopy to return a new object

    def get_by_key_or_none(self, state_key):
        """Get state by key."""
        if self.is_key_valid(state_key):
            state_key = self._convert_state_key_to_string(state_key)
            return copy.deepcopy(self.history[state_key]) # Use deepcopy to return a new object
        else:
            return None

    def _regulate(self):
        """Regulate size of history using max_states."""
        while len(self.history) > self.max_states:
            self.history.popitem(last=True)

    def _convert_state_key_to_string(self, state_key):
        """If not a string it makes it into a string"""
        if not isinstance(state_key, str):
            state_key = str(state_key)
        return state_key

    def int_str_to_int(self, s):
        """Makes a string into an integer if it could be a valid integer"""
        if not isinstance(s, int) and s.isdigit():
            s = int(s)
        return s

    def _validate_nth(self, nth):
        """Validate the nth parameter and raise a ValueError if it's out of range."""
        self._validate_positive_integer(nth)
        if not 1 <= nth <= self.length:
            raise ValueError(f"Invalid position: {nth}. Position should be between 1 and {self.length}.")

    def is_key_valid(self, state_key):
        """Tests if state_key exists in history"""
        self._validate_state_key_not_none(state_key)
        state_key = self._convert_state_key_to_string(state_key)
        keys = self.list_keys()
        return True if state_key in keys else False

    def _validate_state_key_not_none(self, state_key):
        """Validate that a key is not None and raise a ValueError if the key is None"""
        if state_key is None:
            raise ValueError(f"State key was not provided.")
    
    def _validate_state_key_exists(self, state_key):
        """Validate that the state_key exists and raise a ValueError if it doesn't exist in the history."""
        state_key = self._convert_state_key_to_string(state_key)
        keys = self.list_keys()
        if state_key not in keys:
            raise ValueError(f"State key '{state_key}' does not exist in the history.")

    def _validate_positive_integer(self, nth):
        """Validate that nth is a positive integer."""
        if not isinstance(nth, int) or nth <= 0:
            raise ValueError("nth should be a positive integer.")