# Udacity Deep RL ND Multi-Agent Tennis
## Project : Multi-Agent Collaboration & Competition
&nbsp;

## Project Background: Why Multi-agent RL Matters
Artificial intelligence (AI) is on a rapid path to becoming a key component in all the software we build. The software component will rapidly replace human tasks where computers are just better suited to performing. As AI systems become more ubiquitous, these AI systems need to interact safely with humans and other AI systems. AI is already disrupting stock trading and various other practices, and there are future applications that will rely on productive multi-AI agent and human interactions as we drive our world to be more autonomous. Towards this goal of having effective AI-agent and human interaction, reinforcement learning algorithms that train agents in both collaborative and competitive are showing great promise. We have had much success with RL's in single agent domains; these agents act alone and don't need to interact with humans or other AI-agents. These algorithms and approaches (such as Q-Learning) are performing well in single agent mode but are less effective in the complex multi-agent environments since agents are continuously interacting and evolving their policies through interaction and experience.

&nbsp;

## Project Background:
The environment has 2 agents that control rackets to bounce balls over the net. The reward structure for states is as follows:

| States                   | Reward|
|:------------------------:|:-----:|
|    Hit Over the Net      | +0.1  |
| Ball Hits the Ground     | -0.01 |

Thus the reward is constructed to teach the agents to keep the ball in play.

The observation space has 8 variable correspondings to position and velocity of the ball and racket. Each agent receives its local observation 

There are two continuous actions available: 1) moving away and towards the net, 2) jumping. 

The task  is episodic, and the environment is solved when the agents get an average score of +0.50 over 100 consecutive episodes)

## Goal
Train two RL agents to play tennis. Agents must keep the ball in play for as long as possible.

## Getting Started
It is recommended to follow the Udacity DRL ND dependencies [instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies) 

This project utilises [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/) 
A prebuilt simulator is required in be installed. You need only select the environment that matches your operating system:


Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

The file needs to placed in the root directory of the repository and unzipped.

## Instructions
Then run the [`Tennis.ipynb`](https://github.com/bansalayush25/RL_ND_MultiAgentTennis/blob/main/Tennis.ipynb) notebook to train the Multi Agent DDPG agent.

Once trained the model weights will be saved in the same directory in the files `checkpoint_actor_0.pth`, `checkpint_critic_0.pth`, `checkpoint_actor_1.pth` and `checkpint_critic_1.pth`.
