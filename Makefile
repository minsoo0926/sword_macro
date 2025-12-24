init:
	uv sync

train:
	uv run -m rl.train

long-train:
	uv run -m rl.train -t 5000000

long-long-train:
	uv run -m rl.train -t 10000000

test:
	uv run -m rl.test

macro:
	uv run -m macro

tensorboard:
	tensorboard --logdir ./logs/