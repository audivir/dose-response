# dose-response

The script uses the formula for the 4PL-Dose-Response-Curve for fitting:

```
    Y = Bottom + (Top - Bottom) / (1 + 10 ** (HillSlope * (lg EC50 - lg X)))
```

**CAVE: Default unit is µM and confidence level is 0.05 (to change: use the python class manually)**

Input:

- A csv file with doses and its corresponding responses

Output:

- A logarithmic dose-response plot of the fitted curce with standard error bars
- A csv file with the fitted parameters, their SD and the lower and upper bounds of the CI

## Usage

1. Install DRC

```bash
pip install dose-response
```

2. Provide your data results in a CSV file without headers (Mark values to exclude with an * as suffix).

| | |  |  | |
| --- | --- | --- | --- | --- |
| 0.0100 | 0.9648 | 0.9071\* | 1.1355 | 1.0846 |
| 0.0200 | 1.0410 | 1.2995 | 1.0521 | 1.1535 |
| 0.0500 | 1.0967 | 1.1554 | 1.0780 | 1.2259 |
| 0.0100 | 1.2117 | 1.2174 | 1.3170 | 0.9917 |
| 0.0200 | 1.1644 | 1.6411\* | 1.3084 | 1.2987 |
| 0.0500 | 1.4881 | 1.7500 | 1.7305 | 1.3465 |
| 1.0000 | 2.0601 | 2.2339 | 2.2620 | 1.8493 |
| 2.0000 | 1.7824\* | 2.4520\* | 2.1712 | 2.1571 |
| 5.0000 | 2.7854 | 2.6934 | 2.8719 | 2.9488 |
| 10.0000 | 3.1894\* | 2.4637 | 2.3380 | 2.5128 |

3. Execute DRC

   ```
    file: CSV file with data
    -d, --dose-col: Column index of doses: [Default: 0]
    -r, --response-cols: start and end index of response columns: [Default: every other column]
    -o, --out Output directory [Default: current working directory]


    drc DataFile -d DoseColumnIndex -r ResponseStartIndex ResponseEndIndex -o OutputDir

    drc demo/demo.csv -d 0 -r 1 4 -o demo
   ```

1. Plot is stored in OutputDirectory/DataFileBase_plot.png

   ```
    demo/demo_plot.png
   ```

![](https://github.com/tim25651/DRC/blob/main/demo/demo_plot.png?raw=true)

4. Parameters are stored in OutputDirectory/DataFileBase_params.csv

   ```
    demo/demo_params.csv
   ```

| Parameter | Mean | SD | CI_Lower | CI_Upper |
| --- | --- | --- | --- | --- |
| Hill Slope | 1.7123 | 0.3933 | 0.9101 | 2.5144 |
| Top | 2.6591 | 0.0494 | 2.5582 | 2.7599 |
| Bottom | 1.1178 | 0.0855 | 0.9433 | 1.2922 |
| EC50 | 0.7874 | 0.1028 | 0.5778 | 0.9969 |
| LogEC50 | -0.1038 | | -0.2382 | -0.0013 |

## Test

1. Create test data

   ```
    drc-test TestDataFile
   ```

1. Execute DRC

   ```
    drc TestDataFile
   ```

## Acknowledgments

Thanks to yannabraham (https://gist.github.com/yannabraham/5f210fed773785d8b638)
