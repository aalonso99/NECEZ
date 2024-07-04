### Description
Implementation of Neurally Episodically Controlled EfficientZero, a model-based Reinforcement Learning algorithm based on EfficientZero and Neural Episodic Control. This algorithm aims to add interpretability to the model learned by EfficientZero using a parallel database with information about the examples stored in NEC's memory, which are used to evaluate every state observed/simulated by EfficientZero. 

### Running the repository

To test it on, for example, Cartpole, set your configuration in configs/config-cartpole.yaml and run the following commands in terminal.

```
git clone https://github.com/aalonso99/muz.git
cd muz
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py cartpole
```

This will install the virtual environment with the requirements and train an agent with EfficientZero on the Cartpole. Then, you can set the path with the trained model in configs/config-cartpole-nec.yaml and train a NECEZ agent with transfer learning, using the weights of the previously trained EfficientZero agent.

```
python main.py cartpole-nec
```

necez_demo.py implements a simple experiment using a NECEZ agent to simulate, visualize and compute the anomaly score using the NECEZ agent.
