# LSTM

Implementation of LSTM/BLSTM based on Theano.

```python
    # Build Neural Network
    n_input =  84
    n_hidden = n_hidden
    n_output = 1
    
    #model.add(blstm_layer(n_input, n_hidden))
    # BLSTM layer
    for i in range(n_layer):
        model.add(blstm_layer(n_hidden, n_hidden))
    # linear regression layer
    model.add(bi_avec_activate(n_hidden, 1))

    # Choose optimizer
    adadelta = ADADELTA()
    options["optimizer"] = adadelta

    # compile
    model.compile(options)

    # Training
    train_err, valid_err, test_err = model.fit(trainData, validData, testData)
```
