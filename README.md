# graph_rl_tsp

to run python-tsp without dependency errors:
add this as code block to colab (or run w/o ! in terminal)

!pip install --upgrade pip setuptools wheel

!pip uninstall -y numpy tabulate

!pip install "numpy<2.0" "tabulate>=0.9" "networkx>=3.0"
!pip install python-tsp tensordict torch rl4co bigframes gensim langchain matplotlib numba nx-cugraph-cu12 pytensor tensorflow thinc

!pip check

!pip freeze
