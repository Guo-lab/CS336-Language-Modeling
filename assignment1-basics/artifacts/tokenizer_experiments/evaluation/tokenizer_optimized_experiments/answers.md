# Tokenizer Experiments

(a) TinyStories tokenizer: 4.202 bytes/token. OpenWebText tokenizer: 4.694 bytes/token.
(b) The TinyStories tokenizer gets 3.271 bytes/token on the OpenWebText sample, compared with 4.694 bytes/token for the OpenWebText tokenizer; it usually fragments web text more because its vocabulary was learned from simpler story text.
(c) Throughput is about 11,735,567 bytes/second. At that rate, tokenizing 825GB would take about 19.53 hours.
(d) uint16 is appropriate because the 10K and 32K vocabularies both have fewer than 2^16 token IDs, so each token fits in 16 bits while using half the space of uint32.
