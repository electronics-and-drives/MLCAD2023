<h1 align="center">
Differentiable Neural Network Surrogate Models for gm/ID-based Analog IC Sizing
Optimization
</h1>

This repository contains a snapshot of the code base for our MLCAD'23
contribution. It requires a functioning setup of
[libtorch](https://pytorch.org/cppdocs/installing.html), such that
[HaskTorch](http://hasktorch.org/)
can access it. Additionally,
[Serafin](https://github.com/augustunderground/serafin) must
be installed, such that circuits can be found in `~/.serafin`.

## Paper

_Awaiting conference proceedings_

```
@INPROCEEDINGS{UMS23,
    author    = {Uhlmann, Yannick and Moldenhauer, Till and Scheible, J\"{u}rgen},
    booktitle = {2023 {ACM}/{IEEE} 5\textsuperscript{th} Workshop on Machine Learning for CAD (MLCAD)},
    title     = {Differentiable Neural Network Surrogate Models for g\textsubscript{m}/I\textsubscript{D}-based Analog IC Sizing Optimization},
    year      = {2023},
    month     = {09},
    pages     = {1--6},
    doi       = {}
}
```

## Build

Requires `stack`.

```sh
$ git clone https://github.com/electronics-and-drives/MLCAD23
$ cd MCLAD23
$ source setenv
$ stack build
```

## Usage

```sh
Usage: MLCAD23-exe [-m|--mode MODE] [-a|--parallel INT] [-n|--points INT]
                   [-r|--runs INT] [-e|--epochs INT] [-o|--optim OPTIMIZER]
                   [-c|--ckt CKT] [-p|--pdk PDK] [-v|--corner CORNER]
  MLCAD23 Codebase Snapthot

Available options:
  -m,--mode MODE           Mode of operation (default: "Sample")
  -a,--parallel INT        Number of parallel Spectre sessions (default: 64)
  -n,--points INT          Number of data points to sample (default: 50000)
  -r,--runs INT            Number of optimization runs (default: 10)
  -e,--epochs INT          Number of epochs to train for (default: 300)
  -o,--optim OPTIMIZER     Optimizer to use (default: "Numeric")
  -c,--ckt CKT             Circuit ID (default: "sym")
  -p,--pdk PDK             PDK ID (default: "gpdk090")
  -v,--corner CORNER       Process corner (default: "MC")
  -h,--help                Show this help text
```

**Generate Data:**

```sh
$ stack exec -- MLCAD23-exe -m Sample -a 64 -n 50000 -c sym -p gpdk090 -v MC
```

**Train Model:**

```sh
$ stack exec -- MLCAD23-exe -m Train -e 100 -c sym -p gpdk090 -v MC
```

**Optimize Circuit:**

```sh
$ stack exec -- MLCAD23-exe -m Optimize -o Gradient -r 10 -c sym -p gpdk090 -v MC
```
