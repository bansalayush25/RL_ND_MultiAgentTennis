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
