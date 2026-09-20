class BPE:
    def __init__(self, text: str):
        self._text = text
        self._original_tokens = list(map(int, self._text.encode("utf-8")))
        self._merges = {}
        self._vocabulary = {}
        self._BPE_tokens = self._original_tokens

    def _get_stats(self, ids: list) -> dict:
        counts = {}

        for token_index in range(len(ids) - 1):
            current_pair = (ids[token_index], ids[token_index + 1])
            counts[current_pair] = counts.get(current_pair, 0) + 1
            
        return counts

    def _merge(self, previous_tokens, pair, idx):
        new_tokens = []
        skip = False
        for token_index in range(len(previous_tokens)):
            if token_index == len(previous_tokens) - 1:
                new_tokens.append(previous_tokens[token_index])
                break
            
            elif skip == True:
                skip = False
    
            elif previous_tokens[token_index] == pair[0] and previous_tokens[token_index + 1] == pair[1]:
                new_tokens.append(idx)
                skip = True
    
            else:
                new_tokens.append(previous_tokens[token_index])
    
        return new_tokens

    def train(self, vocab_size: int = 276) -> None:
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = self._get_stats(self._BPE_tokens)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            self._BPE_tokens = self._merge(self._BPE_tokens, pair, idx)
            self._merges[pair] = idx

        # building out the vocabulary 
        self._vocabulary = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self._merges.items():
            self._vocabulary[idx] = self._vocabulary[p0] + self._vocabulary[p1]

    def encode(self, text: str) -> list:
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self._merges.get(p, float("inf")))
            if pair not in self._merges:
                break
            idx = self._merges[pair]
            tokens = self._merge(tokens, pair, idx)

        return tokens

    def decode(self, tokens: list ) -> str:
        byte_tokens = []

        for idx in tokens:
            token_bytes = self._vocabulary[idx]
            byte_tokens.append(token_bytes)
    
        all_bytes = b"".join(byte_tokens)
        text = all_bytes.decode("utf-8", errors="replace")
        return text

    @property
    def original_text(self) -> str:
        return self._text

    @property
    def original_tokens(self) -> list:
        return self._original_tokens

    @property
    def legth_of_text(self) -> int:
        return len(self._text)

    @property
    def legth_of_original_tokens(self) -> int:
        return len(self._original_tokens)

    @property
    def merge_statistics(self) -> dict:
        return self._merges

    @property
    def BPE_tokens(self) -> list:
        return self._BPE_tokens

    @property
    def BPE_tokens_length(self) -> list:
        return len(self._BPE_tokens)

    @property
    def vocabulary(self) -> list:
        return self._vocabulary

    @property
    def vocabulary_length(self) -> list:
        return len(self._vocabulary)
    
def main():
    training_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    Test1 = BPE(training_text)

    print(Test1.original_text)
    print(Test1.original_tokens)

    print(Test1.legth_of_text)
    print(Test1.legth_of_original_tokens)

    print(Test1._get_stats(Test1.original_tokens))
    Test1.train(300)
    # print(Test1.BPE_tokens)
    print(Test1.BPE_tokens_length)
    # print(Test1.vocabulary)
    print(Test1.vocabulary_length)

    ecoding_text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to these sheets and more recently with desktop publishing software like Aldus PageMaker and Microsoft Word including versions of Lorem Ipsum."
    BPE_tokens = Test1.encode(ecoding_text)

    print(BPE_tokens)

    print(Test1.decode(BPE_tokens))




if __name__ == "__main__":
    main()