def trainAgent(self, Agent):

    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        observation = self.env.reset()
        total_reward = 0
        while True:
            # self.env.render()

            action = Agent.trainPolicy(observation)

            observation_, reward, done, info = self.env.step(action)

            if done: reward = 10

            Agent.observe(observation, action, reward, observation_, done)

            if total_steps > self.memory_size:
                Agent.learn()
            '''
            if total_steps % 1000 == 0:
                print('\tTotal steps: ', total_steps)
            '''
            total_reward += reward

            if done:
                print('Episode ', i_episode, '\ttotal reward: ', total_reward)
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
