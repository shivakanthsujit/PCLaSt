# Hierarchical Planning

## Setup

```sh
pip install -r req.txt
```

## Data generation

```sh
env="room-multi-passage" # opts are "polygon-obs", "room-multi-passage", "room-spiral", "room-multi-passage-large"

python main.py --env ${env} --opr generate-data
```

## Training PCLaSt encoder

```sh
env="room-multi-passage" # opts are "polygon-obs", "room-multi-passage", "room-spiral", "room-multi-passage-large"

python main.py --env ${env} --opr train --max_k 10 --contrastive --contrastive_k 2 --ndiscrete 64

```

## Clustering

```sh
env="room-multi-passage" # opts are "polygon-obs", "room-multi-passage", "room-spiral", "room-multi-passage-large"

python main.py --env ${env} --opr cluster-latent

```

## Generate Emphirical MDP

```sh
env="room-multi-passage" # opts are "polygon-obs", "room-multi-passage", "room-spiral", "room-multi-passage-large"

python main.py --env ${env} --opr generate-mdp

```

## Excecute plan for goal reaching

```sh
env="room-multi-passage" # opts are "polygon-obs", "room-multi-passage", "room-spiral", "room-multi-passage-large"

python main.py --env ${env} --opr high-low-plan --from_to 2 15
```