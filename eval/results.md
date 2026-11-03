# Sentiment evaluation results

## Custom set (English + Hindi/Hinglish)

### vader

- Accuracy: **0.667**  |  Macro-F1: **0.667**  |  n=45
- Accuracy on `en`: 0.812 (n=32)
- Accuracy on `hi`: 0.308 (n=13)

```
               pred_negative  pred_neutral  pred_positive
true_negative              9             5              3
true_neutral               1            10              1
true_positive              0             5             11
```

### transformer

- Accuracy: **0.956**  |  Macro-F1: **0.956**  |  n=45
- Accuracy on `en`: 0.938 (n=32)
- Accuracy on `hi`: 1.000 (n=13)

```
               pred_negative  pred_neutral  pred_positive
true_negative             16             0              1
true_neutral               1            11              0
true_positive              0             0             16
```
