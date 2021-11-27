# TeachME
#### What makes a Good Teacher?
Modeling Inter-Individual Differences in Humans who Teach Agents Artificial agents (including chatbots, robots, game-playing agents, …) can make use of  interactions with human experts to acquire new skills in a flexible way. A variety of available  algorithms allow human experts to interactively teach agents optimal or near-optimal policies in  dynamic tasks. Depending on what the algorithm allows, the teaching signal can be one or a  combination of evaluative feedback, corrections, demonstrations, advice, etc. (Li et al., 2019,  Celemin et al., 2019, Ravichandar et al. 2020).

Existing research on human-interactive agent learning has focused on designing efficient  agent learners, but largely ignores factors pertaining to the human teachers that can have a direct  or indirect impact on the outcomes of the agent’s learning process. We propose that modeling  inter-individual differences in teaching signals should go hand in hand with designing efficient  algorithms on the agent side, as it could help agents explicitly reason about, adapt, and possibly  influence different types of teachers.

This project is aimed at identifying how features of an individual’s teaching signal, including  the type, timing, accuracy, or frequency of human input, can affect the performance of the agent  as well as the teaching experience of the human. Through a series of studies involving human  participants, we propose to investigate teaching signal variability in interactions between human  teachers and state-of-the-art human-interactive machine learning algorithms, in typical  reinforcement learning benchmark tasks. The output of this research will be a collection of models  capturing inter-individual differences between teachers that can explain different learning  outcomes on the agent side. Such models may unlock new possibilities for designing learning  agents that are more efficient, more flexible, and more human-centered.




##### Human Input Parsing Platform for Openai Gym
The Project is based on HIPPO Gym framework rittwen by Nick Nissen and Yuan Wang Supervised by Matt Taylor and Neda Navi For the Intelligent Robot Learning Laboratory (IRLL) at the University of Alberta (UofA) Supported by the Alberta Machine Intelligence Institure (AMII). Go to https://github.com/IRLL/HIPPO_Gym for more information.


HIPPO Gym is a framework for simplifying human-ai interaction research over the web.
The platform provides a communicator that opens a websocket to pass environment information to a browser-based front-end and recieve commands and actions from the front end. 

Built in Python, the framework is designed as a bring-your-own agent system where the framework provides an easy to impliment middle layer between a researcher's existing agent and a human interacting over the web. The framework is naive in that it makes no assumptions about how human input will be used, it simply provides the mechanism to pass along and record this information.

