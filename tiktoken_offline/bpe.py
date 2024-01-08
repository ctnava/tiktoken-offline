from typing import List, Tuple, Callable, Set
import threading
import re

MAX_NUM_THREADS = 128

def _byte_pair_merge(piece, ranks, func):
    # Args:
    # - piece (bytes): The byte sequence to be merged.
    # - ranks (dict): A dictionary mapping byte pairs to their ranks.
    # - func (callable): A function to apply to the ranges determined by the merge process.
    # 
    # Returns:
    # - list: The result of applying `func` to the ranges.
    parts = [(i, float('inf')) for i in range(len(piece) + 1)]

    def get_rank(parts, start_idx, skip):
        if start_idx + skip + 2 < len(parts):
            byte_pair = piece[parts[start_idx][0]:parts[start_idx + skip + 2][0]]
            return ranks.get(byte_pair, float('inf'))
        return float('inf')

    for i in range(len(parts) - 2):
        rank = get_rank(parts, i, 0)
        if rank != float('inf'):
            parts[i] = (parts[i][0], rank)

    while len(parts) > 1:
        min_rank = (float('inf'), 0)
        for i, (_, rank) in enumerate(parts[:-1]):
            if rank < min_rank[0]:
                min_rank = (rank, i)

        if min_rank[0] != float('inf'):
            i = min_rank[1]
            parts[i] = (parts[i][0], get_rank(parts, i, 1))
            if i > 0:
                parts[i - 1] = (parts[i - 1][0], get_rank(parts, i - 1, 1))
            parts.pop(i + 1)
        else:
            break

    out = []
    for i in range(len(parts) - 1):
        out.append(func(range(parts[i][0], parts[i + 1][0])))

    return out


def hash_current_thread():
    # Generates a hash for the current thread.
    # 
    # Returns:
    # - int: The hash of the current thread ID.
    thread_id = threading.get_ident()
    return hash(thread_id)

class CoreBPE:
    def __init__(
        self,
        encoder: dict,
        special_tokens_encoder: dict,
        pattern: str
    ):
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        special_tokens_pattern = "|".join(map(re.escape, special_tokens_encoder.keys()))
        try:
            special_regex = re.compile(special_tokens_pattern)
        except re.error as e:
            raise ValueError(f"Invalid special tokens regex pattern: {e}")

        # Inverse mapping for encoder
        decoder = {v: k for k, v in encoder.items()}

        # Check for duplicate token indices in encoder
        if len(encoder) != len(decoder):
            raise ValueError("Encoder and decoder lengths do not match; there might be duplicate token indices in the encoder.")

        # Inverse mapping for special_tokens_encoder
        special_tokens_decoder = {v: k.encode() for k, v in special_tokens_encoder.items()}

        # Sorting token bytes
        sorted_token_bytes = sorted(encoder.keys())

        # Initialize class attributes
        self.encoder = encoder
        self.special_tokens_encoder = special_tokens_encoder
        self.decoder = decoder
        self.special_tokens_decoder = special_tokens_decoder
        self.regex_tls = [regex] * MAX_NUM_THREADS  # Define MAX_NUM_THREADS as per your requirement
        self.special_regex_tls = [special_regex] * MAX_NUM_THREADS
        self.sorted_token_bytes = sorted_token_bytes
        
    def byte_pair_encode(self, piece):
        # Encodes a byte sequence using byte pair encoding.
        # 
        # Args:
        # - piece (bytes): The byte sequence to be encoded.
        # - ranks (dict): A dictionary mapping byte pairs to their ranks.
        # 
        # Returns:
        # - list: The ranks corresponding to the encoded byte pairs.
        if len(piece) == 1:
            return [self.encoder[piece]]
        return _byte_pair_merge(piece, self.encoder, lambda p: self.encoder[piece[p.start:p.stop]])

    def byte_pair_split(self, piece):
        # Splits a byte sequence using byte pair encoding.

        # Args:
        # - piece (bytes): The byte sequence to be split.
        # - ranks (dict): A dictionary mapping byte pairs to their ranks.

        # Returns:
        # - list: The byte sequences corresponding to the splits.
        if len(piece) == 1:
            return [piece]
        return _byte_pair_merge(piece, self.encoder, lambda p: piece[p.start:p.stop])

    def _get_tl_regex(self):
        # Gets the thread-local regex.
        # 
        # Returns:
        # - re.Pattern: The regex object for the current thread.
        thread_index = hash_current_thread() % MAX_NUM_THREADS
        return self.regex_tls[thread_index]
    
    def _get_tl_special_regex(self):
        # Gets the thread-local special regex.
        # 
        # Returns:
        # - re.Pattern: The regex object for the current thread.
        thread_index = hash_current_thread() % MAX_NUM_THREADS
        return self.special_regex_tls[thread_index]
    
    def _decode_native(self, tokens):
        # Decodes a sequence of token indices into a sequence of bytes.
        #
        # Args:
        # - tokens (list of int): The token indices to decode.
        # 
        # Returns:
        # - bytes: The decoded byte sequence.
        ret = bytearray()
        for token in tokens:
            token_bytes = self.decoder.get(token, self.special_tokens_decoder.get(token))
            if token_bytes is not None:
                ret.extend(token_bytes)
            else:
                # Handle the case where the token is not found in both decoders
                # You might want to raise an error or handle it in a specific way
                pass
        return bytes(ret)

    def _encode_ordinary_native(self, text):
        # Encodes a string into a sequence of token indices.
        # 
        # Args:
        # - text (str): The text to encode.
        # 
        # Returns:
        # - list of int: The sequence of token indices.
        regex = self._get_tl_regex()
        ret = []
        for match in regex.finditer(text):
            piece = match.group().encode()
            token = self.encoder.get(piece)
            if token is not None:
                ret.append(token)
            else:
                ret.extend(self.byte_pair_encode(piece))
        return ret

    def _encode_native(self, text: str, allowed_special: Set[str]) -> Tuple[List[int], int]:
        # Encodes a string into a sequence of token indices, handling special tokens.
        # 
        # Args:
        # - text (str): The text to encode.
        # - allowed_special (set): A set of allowed special tokens.
        # 
        # Returns:
        # - tuple: A tuple containing the list of token indices and the length of the last piece tokenized.
        special_regex = self._get_tl_special_regex()
        regex = self._get_tl_regex()
        ret = []
        start = 0
        last_piece_token_len = 0

        while True:
            next_special = None
            start_find = start

            while True:
                next_special = special_regex.search(text, start_find)
                if next_special is None:
                    break
                piece = text[next_special.start():next_special.end()]
                if piece in allowed_special:
                    next_special = (next_special.start(), next_special.end())
                    break
                start_find = next_special.end()

            end = next_special[0] if next_special else len(text)

            for mat in regex.finditer(text[start:end]):
                piece = mat.group().encode()
                token = self.encoder.get(piece)
                if token is not None:
                    last_piece_token_len = 1
                    ret.append(token)
                else:
                    tokens = self.byte_pair_encode(piece)
                    last_piece_token_len = len(tokens)
                    ret.extend(tokens)

            if next_special:
                piece = text[next_special[0]:next_special[1]]
                token = self.special_tokens_encoder.get(piece)
                if token is not None:
                    ret.append(token)
                start = next_special[1]
                last_piece_token_len = 0
            else:
                break

        return ret, last_piece_token_len
    
    def _increase_last_piece_token_len(self, tokens, last_piece_token_len):
        # Adjusts the length of the last piece tokenized based on whitespace tokens.
        # 
        # Args:
        # - tokens (list of int): The list of token indices.
        # - last_piece_token_len (int): The length of the last piece tokenized.
        # 
        # Returns:
        # - tuple: A tuple containing the updated list of tokens and the adjusted length.

        def token_is_all_space(token):
            token_bytes = self.decoder.get(token)
            if token_bytes is not None:
                return all(b in [b' ', b'\n', b'\t'] for b in reversed(token_bytes))
            return False

        if last_piece_token_len > 0 and token_is_all_space(tokens[-last_piece_token_len]):
            while last_piece_token_len < len(tokens) and token_is_all_space(tokens[-last_piece_token_len - 1]):
                last_piece_token_len += 1

        assert last_piece_token_len <= len(tokens), "last_piece_token_len should not exceed the length of tokens"

        return tokens, last_piece_token_len
    
    def _encode_unstable_native(self, text, allowed_special):
        tokens, last_piece_token_len = self._encode_native(text, allowed_special)
        if last_piece_token_len == 0:
            return tokens, set()

        tokens, last_piece_token_len = self._increase_last_piece_token_len(tokens, last_piece_token_len)
        unstable_bytes = self._decode_native(tokens[-last_piece_token_len:])
        tokens = tokens[:-last_piece_token_len]

        completions = set()
        if not unstable_bytes:
            return tokens, completions

        # Implement logic for finding completions
        # This includes iterating through sorted_token_bytes and applying additional logic
        # as per the Rust code

        return tokens, completions
    
    # def _byte_pair_encode(self, piece: bytes) -> List[int]:
    #     parts = [(i, float('inf')) for i in range(len(piece) + 1)]

    #     def get_rank(parts: List[Tuple[int, int]], start_idx: int, skip: int):
    #         if start_idx + skip + 2 < len(parts):
    #             start = parts[start_idx][0]
    #             end = parts[start_idx + skip + 2][0]
    #             return self.encoder.get(piece[start:end])
    #         else:
    #             return None

    #     for i in range(len(parts) - 2):
    #         rank = get_rank(parts, i, 0)
    #         if rank is not None:
    #             assert rank != float('inf')
    #             parts[i] = (parts[i][0], rank)

    #     while len(parts) > 1:
    #         min_rank = (float('inf'), 0)
    #         for i, (_, rank) in enumerate(parts[:-1]):
    #             if rank < min_rank[0]:
    #                 min_rank = (rank, i)

    #         if min_rank[0] != float('inf'):
    #             i = min_rank[1]
    #             parts[i] = (parts[i][0], get_rank(parts, i, 1) or float('inf'))
    #             if i > 0:
    #                 parts[i - 1] = (parts[i - 1][0], get_rank(parts))

    # ====================
    # Encoding
    # ====================
    def encode_ordinary(self, text):
        return self._encode_ordinary_native(text)
    
    def encode(self, text: str, allowed_special: Set[str]) -> Tuple[List[int], int]:
        return self._encode_native(text, allowed_special)[0]
    
    def _encode_bytes(self, bytes_data: bytes) -> List[int]:
        """Encodes a byte sequence into a list of token indices.

        Args:
            bytes_data: The byte sequence to encode.

        Returns:
            A list of token indices representing the encoded byte sequence.
        """

        try:
            text = bytes_data.decode('utf-8', errors="ignore")
            return self._encode_ordinary_native(text)
        except UnicodeDecodeError as e:
            valid_text = bytes_data[:e.start].decode('utf-8', errors="ignore")
            tokens, last_piece_token_len = self._encode_native(valid_text, set())
            tokens, last_piece_token_len = self._increase_last_piece_token_len(tokens, last_piece_token_len)
            unstable_bytes = self._decode_native(tokens[-last_piece_token_len:])
            tokens = tokens[:-last_piece_token_len]
            return tokens + [self.encode_single_token(unstable_bytes)]
    
    def encode_with_unstable(self, text, allowed_special):
        # Encodes a string into a sequence of token indices with unstable tokens,
        # returning both the tokens and possible completions.
        # 
        # Args:
        # - text (str): The text to encode.
        # - allowed_special (set): A set of allowed special tokens.
        # 
        # Returns:
        # - tuple: A tuple containing the list of token indices and a list of possible completions.
        tokens, completions = self._encode_unstable_native(text, allowed_special)
        py_completions = [list(seq) for seq in completions]
        return tokens, py_completions
    
    def encode_single_token(self, piece):
        # Encodes a single byte sequence into a token.
        # 
        # Args:
        # - piece (bytes): The byte sequence to encode.
        # 
        # Returns:
        # - int: The encoded token index.
        # 
        # Raises:
        # - KeyError: If the byte sequence cannot be encoded.
        
        # Try encoding using the main encoder
        token = self.encoder.get(piece)
        if token is not None:
            return token

        # Try encoding using the special tokens encoder
        try:
            piece_str = piece.decode('utf-8', errors="ignore")
            token = self.special_tokens_encoder.get(piece_str)
            if token is not None:
                return token
        except UnicodeDecodeError:
            pass

        # If encoding fails, raise a KeyError
        raise KeyError(f"Unable to encode the piece: {piece}")

    def encode_single_piece(self, piece):
        # Encodes a single byte sequence into a list of tokens.
        # 
        # Args:
        # - piece (bytes): The byte sequence to encode.
        # 
        # Returns:
        # - list of int: A list containing the encoded token indices.

        # Check if the piece is directly in the encoder
        token = self.encoder.get(piece)
        if token is not None:
            return [token]

        # If not found, use byte pair encoding
        return self.byte_pair_encode(piece)

    # ====================
    # Decoding
    # ====================
    def decode_bytes(self, tokens):
        return self._decode_native(tokens)
    
    def decode_single_token_bytes(self, token):
        # Decodes a single token index into bytes.
        # 
        # Args:
        # - token (int): The token index to decode.
        # 
        # Returns:
        # - bytes: The decoded byte sequence.
        # 
        # Raises:
        # - KeyError: If the token index cannot be decoded.
        bytes_result = self.decoder.get(token) or self.special_tokens_decoder.get(token)
        if bytes_result is not None:
            return bytes_result
        else:
            raise KeyError(f"Token index {token} cannot be decoded")
        
    # ====================
    # Miscellaneous
    # ====================

    def token_byte_values(self):
        return self.sorted_token_bytes