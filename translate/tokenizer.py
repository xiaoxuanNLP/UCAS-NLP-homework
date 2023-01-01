from .utils import sequence_mask
import nltk


class Tokenizer():
    def __len__(self):
        return

    def encode(self, text, length, pad):
        return

    def decode(self, ids: list, return_list):
        return


class Tokenizer_en(Tokenizer):
    def __init__(self, doc_path):
        self.word2ids_dict = {}
        self.ids2word_dict = {}
        self.prepare_dict(doc_path)

    def vocab_size(self):
        return len(self.word2ids_dict)

    def prepare_dict(self, doc_path):
        with open(doc_path, "r") as f:
            content = f.read()
            words = self.encode(content)
            temp_dict = {}
            for word in words:
                temp_dict[word] = 1
            temp_dict.pop('[CLS]')
            temp_dict.pop('[SEP]')

            # 先加一点特殊符号
            self.word2ids_dict['[PAD]'] = 0
            self.word2ids_dict['[UNK]'] = 1
            self.word2ids_dict['[CLS]'] = 2
            self.word2ids_dict['[SEP]'] = 3
            cursor = 4
            for key in temp_dict.keys():
                self.word2ids_dict[key] = cursor
                cursor += 1

            for key in self.word2ids_dict.keys():
                self.ids2word_dict[self.word2ids_dict[key]] = key

            assert len(self.word2ids_dict) == len(self.ids2word_dict)

    def encode(self, text, length=64, pad=False):
        words = nltk.word_tokenize(text)
        words.insert(0, '[CLS]')

        if length is None:
            words.append('[SEP]')
            mask = sequence_mask(words, length)
            # return ids,mask
        else:
            if len(words) >= length - 1:
                words = words[:length - 1]
                words.append('[SEP]')
                mask = sequence_mask(words, length)
            else:
                words.append('[SEP]')
                mask = sequence_mask(words, length)
                if pad:
                    pads = ['[PAD]' for _ in range(len(words), length, 1)]
                    words.extend(pads)

        ids = []
        for word in words:
            try:
                ids.append(self.word2ids_dict[word])
            except KeyError:
                ids.append(self.word2ids_dict['[UNK]'])

        return ids,mask

    def decode(self, ids, return_list=False):
        decodes = []
        for id in ids:
            if id < 4:
                continue

            decodes.append(self.ids2word_dict[id])

        if return_list:
            return decodes

        sentences = ""
        for word in decodes[:-1]:
            sentences += word + " "
        sentences += decodes[-1] + "."

        return sentences


class Tokenizer_zh(Tokenizer):
    def __init__(self, doc_path):
        self.word2ids_dict = {}
        self.ids2word_dict = {}
        self.prepare_dict(doc_path)

    def vocab_size(self):
        return len(self.word2ids_dict)

    def prepare_dict(self, doc_path):
        with open(doc_path, "r") as f:
            content = f.read()
            words = self.encode(content)
            temp_dict = {}
            for word in words:
                temp_dict[word] = 1
            temp_dict.pop('[CLS]')
            temp_dict.pop('[SEP]')

            # 先加一点特殊符号
            self.word2ids_dict['[PAD]'] = 0
            self.word2ids_dict['[UNK]'] = 1
            self.word2ids_dict['[CLS]'] = 2
            self.word2ids_dict['[SEP]'] = 3
            cursor = 4
            for key in temp_dict.keys():
                self.word2ids_dict[key] = cursor
                cursor += 1

            for key in self.word2ids_dict.keys():
                self.ids2word_dict[self.word2ids_dict[key]] = key

            assert len(self.word2ids_dict) == len(self.ids2word_dict)

    def encode(self, text, length=64, pad=False):
        # words = nltk.word_tokenize(text)
        words = [i if '\u4e00' <= i <= '\u9fff' or '0' <= i <= '9' else "" for i in text]
        words.insert(0, '[CLS]')

        if length is None:
            words.append('[SEP]')
            mask = sequence_mask(words,length)
            # return words,mask
        else:
            if len(words) >= length - 1:
                words = words[:length - 1]
                words.append('[SEP]')
                mask = sequence_mask(words, length)
            else:
                words.append('[SEP]')
                mask = sequence_mask(words, length)
                if pad:
                    pads = ['[PAD]' for _ in range(len(words), length, 1)]
                    words.extend(pads)

        ids = []
        for word in words:
            try:
                ids.append(self.word2ids_dict[word])
            except KeyError:
                ids.append(self.word2ids_dict['[UNK]'])

        return ids,mask

    def decode(self, ids, return_list=False):
        decodes = []
        for id in ids:
            if id < 4:
                continue

            decodes.append(self.ids2word_dict[id])

        if return_list:
            return decodes

        sentences = ""
        for word in decodes[:-1]:
            sentences += word + " "
        sentences += decodes[-1] + "."

        return sentences


if __name__ == "__main__":
    # a = [1,2,3,4]
    # print(a[:-1])

    text = "中国科学院大学400号good"
    text = [i if '\u4e00' <= i <= '\u9fff' or '0' <= i <= '9' else "" for i in text]
    print(text)
