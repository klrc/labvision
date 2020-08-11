import random
import string


def gen_hash(length=8):
    """
        generates a hash with ASCII letters and digits,
        always starts with a letter (for markdown usage).
    """
    assert length > 1
    random.seed()
    _hash_head = ''.join(random.sample(string.ascii_letters, 1))
    _hash_body = ''.join(random.sample(string.ascii_letters+string.digits, length-1))
    return _hash_head+_hash_body
