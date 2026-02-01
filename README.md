# LSTM + Arithmetic Coder compressor

## Description
This project is about a text compression using adaptive arithmetic coding methods + LSTM for predicting symbol sequences. In our modification we use GRU instead of LSTM + we changed hyperparameters. 

BASELINE VS MODIFICATION: <br />

| Version  | Original size, bytes | File size after compression, bytes | Compression ratio | Time, sec |
|----------|----------------------|------------------------------------|-------------------|-----------|
| Baseline | 100000               | 38283                              | 2.63              | 296       |
| Modified | 100000               | 46508                              | 2.15              | 80.77     |

## How to run

```
python fix\main.py
```

## Reference for an arithmetic coder (check their website): 
https://github.com/nayuki/Reference-arithmetic-coding 
https://www.nayuki.io/page/reference-arithmetic-coding
