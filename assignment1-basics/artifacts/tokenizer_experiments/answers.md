# Tokenizer Experiments


(a) TinyStories tokenizer: 4.202 bytes/token. OpenWebText tokenizer: 4.694 bytes/token.

bytes/token 平均每个 token 能表示多少原始 bytes。数字越大，说明压缩越好、切得越粗。

(b) The TinyStories tokenizer gets 3.271 bytes/token on the OpenWebText sample, compared with 4.694 bytes/token for the OpenWebText tokenizer; it usually fragments web text more because its vocabulary was learned from simpler story text.

说明 TinyStories tokenizer 会把 OpenWebText 切得更碎，需要更多 token 表示同样文本

(c) Throughput is about 1,776,099 bytes/second. At that rate, tokenizing 825GB would take about 129.03 hours.

速度大约是每秒处理 1.78MB 文本

(d) uint16 is appropriate because the 10K and 32K vocabularies both have fewer than 2^16 token IDs, so each token fits in 16 bits while using half the space of uint32.
