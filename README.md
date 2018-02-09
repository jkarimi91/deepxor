# DeepXOR

## What is this?
This is my attempt at the XOR problem described in the
[OpenAI Requests for Research 2.0][1],

## What is the problem?
Train an LSTM to solve the `XOR` problem: that is, given a sequence of bits, determine its parity. The [LSTM][2] should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end. Test the two approaches below:

  * Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?
  * Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?


[1]: https://blog.openai.com/requests-for-research-2/
[2]: https://arxiv.org/abs/1706.03762
