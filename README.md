Citation information: 
Spangher, et al. ``Benchmarking models in a test suite for disruptions warnings." 2024. Pre-print. (Include URL.) 

**DisruptionBench**: Evaluation class for disruption prediction

Please see the associated notebook, model_performance.ipynb, for an example of how to use the class; indeed, if using within your own disruption predictor you may simply copy the notebok and input your own datafiles. 

Please pay careful attention to the dimensionality of the inputs expected; you may need an np.unsqueeze(-1) or reshape command to put an extra dimension at the end of your data. 

The main workhorse of the benchmark is the function: 

```# Compute metrics_report_
val_metrics_report = performance_calculator.eval(
    unrolled_proba = dict_y_proba_shots_val,
    metrics = metrics,
    params_dict = params_dict
)
```

In order to use this, you must massage your disruptivity scores into the dictionary dict_y_proba_shots_val. The four fields are noted.

Some notes:
- You may construct the field ``time until disruption'' by merely subtracting the your time slices' indices from the length of the sequence and multiplying by your sampling resolution, i.e.: 

``` 
(len(observation) - np.linspace(0, len(observation)) * sampling_frequency
```

- the Time shot is an incrementing count of the sampling frequency. If you are only using shots with the same time base, i.e. .005 seconds throughout, this column can simply be incrementing ones.

The ``params'' dictionary allows you to specify hyperparameters like thresholds and hysteresis. The fields have the following meaning:
  - 'high_thr': the high threshold above which a disruption warning system counts. 
  - 'low_thr': a lower threshold; if above this threshold but below the high threshold, the disruption warning system does not count, and if below, it resets. 
  - 't_hysteresis': the number of time-steps above the higher threshold needed before an alarm is triggered. 
  - 't_useful': the amount of time prior to the end of a shot that the disruption warning system is useful. I.e., if a positive is flagged after this metric, it will not be considered a valid positive. 
