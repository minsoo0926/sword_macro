train:
	python -m rl.train

long-train:
	python -m rl.train -t 5000000

long-long-train:
	python -m rl.train -t 10000000

test:
	python -m rl.test

macro:
	python -m macro

tensorboard:
	tensorboard --logdir ./logs/