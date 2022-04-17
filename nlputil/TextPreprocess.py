import collections

from nlputil import Crawler


def tokenize(lines, token_type='word', dim=2):
    if token_type == 'word':
        res = [line.split() for line in lines]
        if dim == 2:
            return res
        elif dim == 1:
            new_res = []
            for words in res:
                for word in words:
                    new_res.append(word)
            return new_res
    elif token_type == 'char':
        if dim == 2:
            return [list(line) for line in lines]
        elif dim == 1:
            return [c for line in lines for c in list(line)]
    else:
        return []


class Vocabulary:
    """
    文本词表
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # 统计词频得到词频字典
        if reserved_tokens is None: reserved_tokens = []
        if tokens is None: tokens = []
        freqs = self.count_freq(tokens)
        # 词频字典进行二元组排序，即[(s,cnt[s])]进行排序
        self.sort_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        # 抛弃词或者保留字词先初始化id
        self.idx2token = ['<unk>'] + reserved_tokens
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        # 对不是抛弃词的词语建立到数字索引的建立
        for token, freq in self.sort_freqs:
            if freq < min_freq: break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, item):
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx[tokens]
        return [self.__getitem__(token) for token in tokens]

    def idxs2tokens(self, indexs):
        if not isinstance(indexs, (list, tuple)):
            return self.idx2token[indexs]
        return [self.idxstotokens(index) for index in indexs]

    def count_freq(self, tokens):  # @save
        """统计词元的频率"""
        # 这⾥的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成⼀个列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):
        return self.token2idx['unk']

    @property
    def freqs(self):
        return self.freqs


if __name__ == '__main__':
    # 1.下载读取字符串数据
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/' + 'timemachine.txt'
    crawler = Crawler(url)
    text = crawler.download_from_url()
    lines = text.split('\n')
    # 2.词元化
    tokens = tokenize(lines, 'word', 1)
    # 3.词表
    voc = Vocabulary(tokens)
    print(voc.idx2token[:10])
