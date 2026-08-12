import string


class CharTokenizer:
    PAD, SEP, EOS = 0, 1, 2

    def __init__(self):
        # 10 digits + 52 letters + 32 punctuation + 1 space = 95
        alphabet = string.digits + string.ascii_letters + string.punctuation + " "
        self.itos = ["<pad>", "<sep>", "<eos>"] + list(alphabet)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character not in vocab: {e}") from None

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i > self.EOS)

    def encode_pair(self, question, answer):
        return self.encode(question) + [self.SEP] + self.encode(answer) + [self.EOS]