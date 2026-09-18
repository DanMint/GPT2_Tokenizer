class BPE:
    def __init__(self, text: str):
        self._text = text
        self._original_tokens = list(map(int, self._text.encode("utf-8")))
        self._merges = {}
        self._vocabulary = {}
        self._BPE_tokens = self._original_tokens

    def get_stats(self, ids: list) -> dict:
        counts = {}

        for token_index in range(len(ids) - 1):
            current_pair = (ids[token_index], ids[token_index + 1])
            counts[current_pair] = counts.get(current_pair, 0) + 1
            
        return counts

    def merge(self, previous_tokens, pair, idx):
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

    def train(self, vocab_size: int = 276):
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(self._BPE_tokens)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            self._BPE_tokens = self.merge(self._BPE_tokens, pair, idx)
            self._merges[pair] = idx

        # building out the vocabulary 
        self._vocabulary = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self._merges.items():
            self._vocabulary[idx] = self._vocabulary[p0] + self._vocabulary[p1] 

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
    Test1 = BPE("Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.")

    print(Test1.original_text)
    print(Test1.original_tokens)

    print(Test1.legth_of_text)
    print(Test1.legth_of_original_tokens)

    print(Test1.get_stats(Test1.original_tokens))
    Test1.train(300)
    # print(Test1.BPE_tokens)
    print(Test1.BPE_tokens_length)
    # print(Test1.vocabulary)
    print(Test1.vocabulary_length)

if __name__ == "__main__":
    main()