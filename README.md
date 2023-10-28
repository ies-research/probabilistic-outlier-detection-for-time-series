# PrOuD: Probabilistic Outlier Detection for Time Series
This repository contains Python code to implement the PrOuD (Probabilistic Outlier Detection) framework applied to the experiments the paper "Probabilistic Outlier Detection for Time Series".

## Code Structure

```
DTS
├───anomaly_conf             #anomaly_conf, data_conf, model_conf
├───utils                    #framework implementation
├───visualization            #visualization of detection result for real/sytecti dataset
├───warmup_models            #The trained prediction model in warmup phase
│   
├── README.md
│   
├── artficial_time_series.ipynb #The notebook contains the theory and code of artificial time series generation; the principle and code of adding artificial anomalies.
│          
└── main.py                  #The entry of code
    
```

# Useage:
1. cd to root of the code
2. python main.py --num-inverter=artificial_dataset --synthetic-anomaly-type=m --train-methods=MCDHDNN

| Parameter                | value  | description|
| ------------------------ | ------ | -----------|
| `--num-inverter`         | [1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, artificial_dataset] | real num of inverter data or artificial_dataset |
| `--synthetic-anomaly-type` | [1,2,3,4,m] |  for different novel pattern, m means mixed types|
| `--train-methods`          | [MCDHDNN, VEHDNN, VEHLSTM] | different prediction models|




